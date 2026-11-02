"""The Cover tab: the whole pipeline in one click."""

import logging
from dataclasses import asdict, dataclass, fields, replace

import gradio as gr

from ai_cover_maker.core.downloads.ytdl import download_audio
from ai_cover_maker.core.effects.chain import VocalEffects
from ai_cover_maker.core.pipeline.options import OUTPUT_FORMATS, SEPARATION_STAGES, CoverOptions, load_presets, preset_options
from ai_cover_maker.core.pipeline.runner import CoverPipeline
from ai_cover_maker.core.rvc.engine import CLEANUP_MODES, EMBEDDERS, F0_METHODS, RvcParams
from ai_cover_maker.core.separation.service import clean_path, is_folder
from ai_cover_maker.i18n import N_, i18n
from ai_cover_maker.integrations.discord_presence import presence
from ai_cover_maker.ui import a11y, components
from ai_cover_maker.ui.adapters import (GPU_CONCURRENCY_ID, filled, progress_callback, require, run_streaming,
                                        user_errors, validations)
from ai_cover_maker.ui.context import AppContext
from ai_cover_maker.ui.rvc_controls import refresh_voices, rvc_params, rvc_settings, voice_dropdown, voice_names

logger = logging.getLogger(__name__)

STAGE_LABELS = {
    "vocals": N_("Vocals / instrumental model"),
    "karaoke": N_("Lead / backing vocals model"),
    "dereverb": N_("De-reverb model"),
    "deecho": N_("De-echo model"),
    "denoise": N_("Denoise model"),
}
STAGE_INFO = {
    "vocals": N_("Separates the vocals from the instrumental"),
    "karaoke": N_("Optional. Converts only the lead vocals and keeps the backing vocals apart"),
    "dereverb": N_("Optional. RVC sounds much cleaner on dry vocals"),
    "deecho": N_("Optional. Removes echo/delay from the lead vocals"),
    "denoise": N_("Optional. Removes noise from the lead vocals"),
}
EFFECT_FIELDS = {f.name for f in fields(VocalEffects)}
DEFAULT_PRESET = "Balanced"


@dataclass
class CoverTab:
    voice: gr.Dropdown
    backing_voice: gr.Dropdown
    settings: dict[str, gr.components.Component]  # what "Save settings" stores, by default_settings() name


def default_settings(presets: dict) -> dict:
    """Every Cover setting the user can save, by handler parameter name, at its default value."""
    stages = preset_options(presets[DEFAULT_PRESET])
    rvc = {k: v for k, v in asdict(RvcParams()).items() if k != "sid"}
    rvc["embedder_model_custom"] = rvc["embedder_model_custom"] or ""
    names = voice_names()
    return {
        "voice": names[0] if names else None, "output_format": "wav", "preset": DEFAULT_PRESET, "all_models": False,
        **{f"{stage}_model": stages[f"{stage}_model"] or "" for stage in SEPARATION_STAGES},
        **rvc, "backing_voice": "", "backing_pitch": 0, **asdict(VocalEffects()),
        "key_shift": 0, "lead_gain_db": 0.0, "backing_gain_db": 0.0, "instrumental_gain_db": 0.0,
    }


def initial_settings(saved: dict, defaults: dict, choices: dict[str, list]) -> dict:
    """``defaults`` with the saved values that are still valid (a saved voice or model may be gone)."""
    values = dict(defaults)
    for name, value in saved.items():
        if name not in defaults or (name in choices and value not in choices[name]):
            continue
        default = defaults[name]
        if default is None or _same_kind(value, default):
            values[name] = value
        else:
            logger.warning("Ignoring the saved Cover setting %s=%r", name, value)
    return values


def _same_kind(value, default) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(default, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float))
    return isinstance(value, type(default))


def render_cover_tab(ctx: AppContext) -> CoverTab:
    presets = load_presets()
    saved = ctx.cover_settings.load()
    # A saved model outside its step is only valid while "Show all models" is on
    saved_all = saved.get("all_models") is True
    voices = voice_names()
    choices = {"voice": voices, "backing_voice": ["", *voices], "preset": list(presets),
               "output_format": OUTPUT_FORMATS, "f0_method": F0_METHODS, "embedder_model": EMBEDDERS,
               "cleanup_mode": CLEANUP_MODES,
               **{f"{stage}_model": ["", *stage_model_ids(ctx.catalog, stage, saved_all)] for stage in SEPARATION_STAGES}}
    initial = initial_settings(saved, default_settings(presets), choices)

    with gr.TabItem(i18n("Cover")):
        a11y.tab_heading(i18n("Cover"))
        # equal_height: buttons as tall as the dropdowns and textboxes next to them
        with gr.Row(equal_height=True):
            voice = voice_dropdown(value=initial["voice"])
            refresh = gr.Button(i18n("Refresh voice models"), variant="primary", scale=0)
            output_format = gr.Dropdown(OUTPUT_FORMATS, value=initial["output_format"], label=i18n("Select the output format"),
                                        info=i18n("Of the cover and of every stem saved with it"))
        with gr.Row():
            audio = gr.Audio(label=i18n("Song"), type="filepath", interactive=True)
        with gr.Accordion(i18n("Download a song by link"), open=False):
            with gr.Row(equal_height=True):
                link = components.single_line_textbox(i18n("Link"), i18n("Paste the link here"))
                download = gr.Button(i18n("Download!"), variant="primary", scale=0)
            gr.Markdown(i18n("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)"))

        with gr.Accordion(i18n("Separation steps"), open=False):
            with gr.Row(equal_height=True):
                preset = gr.Dropdown(list(presets), value=initial["preset"], label=i18n("Preset"),
                                     info=i18n(presets[initial["preset"]].get("description", "")), interactive=True,
                                     scale=3)
                all_models = gr.Checkbox(value=initial["all_models"], label=i18n("Show all models"),
                                         info=i18n("Lists every model in each step, not only the ones made for it"),
                                         scale=1)
            stages = {}
            for stage in SEPARATION_STAGES:
                stages[stage] = gr.Dropdown(stage_choices(ctx.catalog, stage, initial["all_models"]),
                                            value=initial[f"{stage}_model"], label=i18n(STAGE_LABELS[stage]),
                                            info=i18n(STAGE_INFO[stage]), interactive=True)

        with gr.Accordion(i18n("Voice conversion settings"), open=False):
            rvc = rvc_settings(rvc_params(initial))

        with gr.Accordion(i18n("Backing vocals"), open=False):
            gr.Markdown(i18n("Needs a lead / backing vocals model. Without a voice here the backing vocals keep the original singer"))
            with gr.Row():
                backing_voice = gr.Dropdown(choices=backing_choices(), value=initial["backing_voice"], label=i18n("Backing vocals voice model"),
                                            interactive=True)
                backing_pitch = gr.Slider(-24, 24, value=initial["backing_pitch"], step=1, label=i18n("Backing vocals pitch"))

        with gr.Accordion(i18n("Vocal effects"), open=False):
            effects = _effects_controls(VocalEffects(**{k: initial[k] for k in EFFECT_FIELDS}))

        with gr.Accordion(i18n("Mix"), open=False):
            with gr.Row():
                key_shift = gr.Slider(-12, 12, value=initial["key_shift"], step=1, label=i18n("Change the key of the song"),
                                      info=i18n("Semitones for the whole song: instrumental and vocals, keeping the tempo"))
            with gr.Row():
                lead_gain = gr.Slider(-20, 10, value=initial["lead_gain_db"], step=0.5, label=i18n("Lead vocals volume (dB)"))
                backing_gain = gr.Slider(-20, 10, value=initial["backing_gain_db"], step=0.5, label=i18n("Backing vocals volume (dB)"))
                instrumental_gain = gr.Slider(-20, 10, value=initial["instrumental_gain_db"], step=0.5, label=i18n("Instrumental volume (dB)"))

        with gr.Accordion(i18n("Batch covers"), open=False):
            with gr.Row():
                input_path = components.single_line_textbox(i18n("Input path"), i18n("Place the input path here"))
                output_path = components.single_line_textbox(i18n("Output path"), i18n("Place the output path here"))
            with gr.Row():
                batch_button = gr.Button(i18n("Make all the covers!"), variant="primary")
                stop_button = gr.Button(i18n("Stop"), variant="stop")
            batch_info = components.output_log()

        with gr.Row():
            make = gr.Button(i18n("Make cover!"), variant="primary")
        cover = gr.Audio(label=i18n("Cover"), type="filepath", interactive=False, buttons=["download"])
        with gr.Row():
            instrumental = gr.Audio(label=i18n("Instrumental"), type="filepath", interactive=False, buttons=["download"])
            backing = gr.Audio(label=i18n("Backing vocals"), type="filepath", interactive=False, buttons=["download"],
                               visible=False)
        log = components.output_log()
        announcer = a11y.announcer()

    stage_inputs = {f"{stage}_model": stages[stage] for stage in SEPARATION_STAGES}
    option_inputs = {
        "voice": voice, **stage_inputs, **rvc, **effects, "backing_voice": backing_voice, "backing_pitch": backing_pitch,
        "key_shift": key_shift, "lead_gain_db": lead_gain, "backing_gain_db": backing_gain,
        "instrumental_gain_db": instrumental_gain, "output_format": output_format,
    }

    def build_options(values: dict) -> CoverOptions:
        params = rvc_params(values)
        return CoverOptions(
            voice=require(values["voice"], N_("Please select a voice model")),
            **{f"{stage}_model": values[f"{stage}_model"] or None for stage in SEPARATION_STAGES},
            rvc=params,
            backing_voice=values["backing_voice"] or None,
            backing_rvc=replace(params, pitch=int(values["backing_pitch"])),
            effects=VocalEffects(**{k: v for k, v in values.items() if k in EFFECT_FIELDS}),
            key_shift=int(values["key_shift"]),
            lead_gain_db=values["lead_gain_db"], backing_gain_db=values["backing_gain_db"],
            instrumental_gain_db=values["instrumental_gain_db"],
            output_format=values["output_format"] or "wav",
        )

    def pipeline(output_dir=None) -> CoverPipeline:
        # Separation parameters and precision are always the defaults
        return CoverPipeline(catalog=ctx.catalog, outputs_dir=output_dir)

    @user_errors
    def run(audio, progress=gr.Progress(), **values):
        song = require(audio, N_("Please provide an input audio"))
        options = build_options(values)

        def work(emit, report):
            return pipeline().run(song, options, progress=report, on_log=emit)

        # The log shows each step as it happens; the progress bars are on the audio players
        with presence.track("cover"):
            result = yield from run_streaming(work, render=lambda lines: (gr.skip(), gr.skip(), gr.skip(),
                                                                          "\n".join(lines)),
                                              progress=progress_callback(progress))
        backing_path = result.stems.get("backing")
        yield (str(result.cover), str(result.stems["instrumental"]),
               gr.update(value=str(backing_path) if backing_path else None, visible=backing_path is not None),
               "\n".join(result.log))

    # Handler parameters are the API's names ("audio", "input_path"), so they shadow the components
    audio_input, input_path_box = audio, input_path

    def validate(audio, **values):
        return validations(
            (voice, filled(values["voice"]), N_("Please select a voice model")),
            (audio_input, filled(audio), N_("Please provide an input audio")),
            (stages["vocals"], filled(values["vocals_model"]), N_("Please select a model first")),
        )

    @user_errors
    def run_batch(input_path, output_path, progress=gr.Progress(), **values):
        options = build_options(values)
        lines = []
        with presence.track("cover", batch=True):
            for line in pipeline(clean_path(output_path) or None).run_batch(clean_path(input_path), options,
                                                                            progress=progress_callback(progress)):
                lines.append(line)
                yield "\n".join(lines)

    def validate_batch(input_path, output_path, **values):
        return validations(
            (voice, filled(values["voice"]), N_("Please select a voice model")),
            (input_path_box, is_folder(input_path), N_("The input path must be an existing folder")),
        )

    def apply_preset(name):
        chosen = presets.get(name) or {}
        kwargs = preset_options(chosen)
        return [gr.update(info=i18n(chosen.get("description", "")))] + [
            kwargs[f"{stage}_model"] or "" for stage in SEPARATION_STAGES]

    def show_all_models(show_all, preset_name, *selected):
        """Each step's dropdown with its own models or the whole catalog, keeping the selections that still fit."""
        chosen = preset_options(presets.get(preset_name) or {})
        updates = []
        for stage, value in zip(SEPARATION_STAGES, selected):
            ids = stage_model_ids(ctx.catalog, stage, show_all)
            fallback = chosen[f"{stage}_model"] or ""
            updates.append(gr.update(choices=stage_choices(ctx.catalog, stage, show_all),
                                     value=value if value in ids or value == "" else fallback))
        return updates

    preset.change(apply_preset, inputs=[preset], outputs=[preset, *stages.values()], api_visibility="undocumented")
    all_models.change(show_all_models, inputs=[all_models, preset, *stages.values()], outputs=list(stages.values()),
                      show_progress="hidden", api_visibility="undocumented")
    refresh.click(refresh_cover_voices, inputs=[voice, backing_voice], outputs=[voice, backing_voice],
                  api_visibility="undocumented")
    download.click(user_errors(lambda url: str(download_audio(url))), inputs=[link], outputs=[audio],
                   api_name="download_song")

    # Gradio draws a progress bar only over an empty player, and not over one the running event
    # has updated, so the previous cover is cleared in the browser before the cover starts
    clear = make.click(None, js="() => [null, null, null]", outputs=[cover, instrumental, backing],
                       api_visibility="undocumented")
    event = clear.then(run, inputs_kwargs={"audio": audio, **option_inputs},
                       outputs=[cover, instrumental, backing, log], validator=validate, api_name="make_cover",
                       show_progress_on=[cover, instrumental, backing], concurrency_limit=1,
                       concurrency_id=GPU_CONCURRENCY_ID)
    event.success(lambda: a11y.announce(i18n("Cover finished")), outputs=announcer, show_progress="hidden",
                  api_visibility="undocumented")
    batch_event = batch_button.click(run_batch, inputs_kwargs={"input_path": input_path, "output_path": output_path,
                                                               **option_inputs},
                                     outputs=[batch_info], validator=validate_batch, api_name="make_covers_batch",
                                     concurrency_limit=1, concurrency_id=GPU_CONCURRENCY_ID)
    stop_button.click(None, cancels=[batch_event], api_visibility="undocumented")
    # Saved from the Settings tab: every option of the cover plus the preset and "Show all models"
    return CoverTab(voice=voice, backing_voice=backing_voice,
                    settings={"preset": preset, "all_models": all_models, **option_inputs})


def stage_model_ids(catalog, stage: str, show_all: bool) -> list[str]:
    """The ids of the models a step lists: the ones tagged for it, or every model."""
    return [m.id for m in (catalog if show_all else catalog.for_stage(stage))]


def stage_choices(catalog, stage: str, show_all: bool) -> list[tuple[str, str]]:
    """A step's dropdown choices: "Skip this step" for the optional steps, then its models, recommended first."""
    models = list(catalog) if show_all else catalog.for_stage(stage)
    skip = [(i18n("Skip this step"), "")] if stage != "vocals" else []
    return skip + components.model_choices(catalog, models)


def backing_choices() -> list[tuple[str, str]]:
    return [(i18n("Keep the original backing vocals"), "")] + [(name, name) for name in voice_names()]


def refresh_cover_voices(voice, backing_voice):
    """New voice lists for the lead and backing dropdowns, keeping the selections that still exist."""
    choices = backing_choices()
    backing = backing_voice if backing_voice in [value for _, value in choices] else ""
    return refresh_voices(voice), gr.update(choices=choices, value=backing)


def _effects_controls(d: VocalEffects) -> dict:
    c = {}
    with gr.Row():
        c["highpass"] = gr.Checkbox(value=d.highpass, label=i18n("High-pass filter"), info=i18n("Removes rumble below the cutoff"))
        c["highpass_hz"] = gr.Slider(20, 300, value=d.highpass_hz, step=1, label=i18n("Cutoff frequency (Hz)"))
    with gr.Row():
        c["compressor"] = gr.Checkbox(value=d.compressor, label=i18n("Compressor"), info=i18n("Evens out the vocal volume"))
        c["compressor_threshold_db"] = gr.Slider(-60, 0, value=d.compressor_threshold_db, step=1, label=i18n("Threshold (dB)"))
        c["compressor_ratio"] = gr.Slider(1, 20, value=d.compressor_ratio, step=0.5, label=i18n("Ratio"))
    with gr.Row():
        c["reverb"] = gr.Checkbox(value=d.reverb, label=i18n("Reverb"), info=i18n("Puts the dry converted vocals back in a room"))
        c["reverb_room_size"] = gr.Slider(0, 1, value=d.reverb_room_size, step=0.01, label=i18n("Room size"))
        c["reverb_damping"] = gr.Slider(0, 1, value=d.reverb_damping, step=0.01, label=i18n("Damping"))
    with gr.Row():
        c["reverb_wet_level"] = gr.Slider(0, 1, value=d.reverb_wet_level, step=0.01, label=i18n("Wet level"))
        c["reverb_dry_level"] = gr.Slider(0, 1, value=d.reverb_dry_level, step=0.01, label=i18n("Dry level"))
        c["reverb_width"] = gr.Slider(0, 1, value=d.reverb_width, step=0.01, label=i18n("Width"))
    # An effect's settings only apply while it is checked
    for effect, settings in (("highpass", ["highpass_hz"]),
                             ("compressor", ["compressor_threshold_db", "compressor_ratio"]),
                             ("reverb", ["reverb_room_size", "reverb_damping", "reverb_wet_level",
                                         "reverb_dry_level", "reverb_width"])):
        outputs = [c[name] for name in settings]
        for component in outputs:
            component.interactive = getattr(d, effect)
        # A dict, not a list: with a single output Gradio would take the list as the slider's value
        c[effect].change(lambda on, outs=outputs: {o: gr.update(interactive=on) for o in outs}, inputs=[c[effect]],
                         outputs=outputs, show_progress="hidden", api_visibility="undocumented")
    return c
