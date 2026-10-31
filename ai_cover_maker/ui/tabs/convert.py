"""The Voice conversion tab: RVC on vocals the user already has (no separation)."""

import tempfile
from pathlib import Path

import gradio as gr

from ai_cover_maker import paths
from ai_cover_maker.core.audio import list_audio_files
from ai_cover_maker.core.rvc.engine import rvc_engine
from ai_cover_maker.core.rvc.voices import get_voice
from ai_cover_maker.core.separation.service import clean_path, is_folder
from ai_cover_maker.i18n import N_, i18n
from ai_cover_maker.ui import a11y, components
from ai_cover_maker.ui.adapters import GPU_CONCURRENCY_ID, filled, progress_callback, require, user_errors, validations
from ai_cover_maker.ui.rvc_controls import refresh_voices, rvc_params, rvc_settings, voice_dropdown


def _voice(name):
    require(name, N_("Please select a voice model"))
    try:
        return get_voice(name)
    except KeyError:
        raise gr.Error(i18n("Voice model not found: {voice}").format(voice=name))


def render_convert_tab() -> gr.Dropdown:
    with gr.TabItem(i18n("Voice conversion")):
        a11y.tab_heading(i18n("Voice conversion"))
        gr.Markdown(i18n("Convert vocals you already have. To start from a full song use the Cover tab"))
        with gr.Row():
            voice = voice_dropdown()
            refresh = gr.Button(i18n("Refresh voice models"), variant="secondary", scale=0)
        with gr.Row():
            audio = gr.Audio(label=i18n("Input vocals"), type="filepath", interactive=True)
        rvc = rvc_settings()
        with gr.Accordion(i18n("Batch conversion"), open=False):
            with gr.Row():
                input_path = components.single_line_textbox(i18n("Input path"), i18n("Place the input path here"))
                output_path = components.single_line_textbox(i18n("Output path"), i18n("Place the output path here"))
            with gr.Row():
                batch_button = gr.Button(i18n("Convert all files!"), variant="primary")
                stop_button = gr.Button(i18n("Stop"), variant="stop")
            batch_info = components.output_log()
        with gr.Row():
            convert = gr.Button(i18n("Convert!"), variant="primary")
        converted = gr.Audio(label=i18n("Converted vocals"), type="filepath", interactive=False, buttons=["download"])

    # Handler parameters are the API's names (voice, audio, input_path), so they shadow the components
    voice_input, audio_input, input_path_box = voice, audio, input_path

    @user_errors
    def run(voice, audio, progress=gr.Progress(), **values):
        source = Path(require(audio, N_("Please provide an input audio")))
        out = Path(tempfile.mkdtemp(prefix="ai_cover_maker-rvc-")) / f"{source.stem} ({voice.replace('/', ' ')}).wav"
        return str(rvc_engine.convert(source, out, _voice(voice), rvc_params(values), progress_callback(progress)))

    def validate(voice, audio, **values):
        return validations(
            (voice_input, filled(voice), N_("Please select a voice model")),
            (audio_input, filled(audio), N_("Please provide an input audio")),
        )

    @user_errors
    def run_batch(voice, input_path, output_path, progress=gr.Progress(), **values):
        selected, params = _voice(voice), rvc_params(values)
        output_dir = Path(clean_path(output_path) or paths.OUTPUTS_DIR)
        files = list_audio_files(Path(clean_path(input_path)))
        lines = [f"{len(files)} audio files found"]
        yield "\n".join(lines)
        for i, file in enumerate(files):
            progress(i / max(len(files), 1), desc=f"{i + 1}/{len(files)}: {file.name}")
            try:
                out = rvc_engine.convert(file, output_dir / f"{file.stem} ({voice.replace('/', ' ')}).wav", selected, params)
                lines.append(f"Done: {out.name}")
            except Exception as e:
                lines.append(f"Error with {file.name}: {e}")
            yield "\n".join(lines)
        lines.append(f"Finished. Output: {output_dir}")
        yield "\n".join(lines)

    def validate_batch(voice, input_path, output_path, **values):
        return validations(
            (voice_input, filled(voice), N_("Please select a voice model")),
            (input_path_box, is_folder(input_path), N_("The input path must be an existing folder")),
        )

    refresh.click(refresh_voices, inputs=[voice_input], outputs=[voice_input], api_visibility="undocumented")
    convert.click(run, inputs_kwargs={"voice": voice_input, "audio": audio_input, **rvc}, outputs=[converted], validator=validate,
                  api_name="convert_voice", concurrency_limit=1, concurrency_id=GPU_CONCURRENCY_ID)
    batch_event = batch_button.click(run_batch, inputs_kwargs={"voice": voice_input, "input_path": input_path_box,
                                                               "output_path": output_path, **rvc},
                                     outputs=[batch_info], validator=validate_batch, api_name="convert_voice_batch",
                                     concurrency_limit=1, concurrency_id=GPU_CONCURRENCY_ID)
    stop_button.click(None, cancels=[batch_event], api_visibility="undocumented")
    return voice_input
