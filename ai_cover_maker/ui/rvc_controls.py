"""Widgets shared by the tabs that run RVC: voice selector and conversion settings."""

from dataclasses import fields

import gradio as gr

from ai_cover_maker.core.rvc.engine import CLEANUP_MODES, EMBEDDERS, F0_METHODS, RvcParams
from ai_cover_maker.core.rvc.voices import list_voices
from ai_cover_maker.i18n import i18n

RVC_FIELDS = {f.name for f in fields(RvcParams)}


def voice_names() -> list[str]:
    return [v.name for v in list_voices()]


def voice_dropdown(label: str = None, value: str = None) -> gr.Dropdown:
    names = voice_names()
    return gr.Dropdown(label=label or i18n("Voice model"), choices=names,
                       value=value if value in names else (names[0] if names else None),
                       interactive=True, info=i18n("Voice models are kept in models/rvc. Add more in the Voices tab"))


def refresh_voices(*current):
    """``gr.update`` for every voice dropdown, keeping each selection when it still exists."""
    names = voice_names()
    updates = [gr.update(choices=names, value=value if value in names else (names[0] if names else None))
               for value in current]
    return updates[0] if len(updates) == 1 else updates


def rvc_settings(defaults: RvcParams = RvcParams()) -> dict[str, gr.components.Component]:
    """Conversion settings keyed by ``RvcParams`` field name (except ``sid``)."""
    c = {}
    with gr.Row():
        c["pitch"] = gr.Slider(-24, 24, value=defaults.pitch, step=1, label=i18n("Pitch"),
                               info=i18n("Semitones. +12 for a male voice to a female model, -12 for the opposite"))
        c["f0_method"] = gr.Dropdown(F0_METHODS, value=defaults.f0_method, label=i18n("Pitch extraction method"),
                                     info=i18n("rmvpe is the best choice for singing"))
    with gr.Row():
        c["index_rate"] = gr.Slider(0, 1, value=defaults.index_rate, step=0.01, label=i18n("Search feature ratio"),
                                    info=i18n("Influence of the index file: more accent of the model, but more artifacts"))
        c["protect"] = gr.Slider(0, 0.5, value=defaults.protect, step=0.01, label=i18n("Protect voiceless consonants"),
                                 info=i18n("Keeps breaths and consonants clean. 0.5 disables it"))
        c["volume_envelope"] = gr.Slider(0, 1, value=defaults.volume_envelope, step=0.01, label=i18n("Volume envelope"),
                                         info=i18n("1 keeps the loudness of the converted voice, 0 follows the original singer"))
    with gr.Accordion(i18n("More conversion settings"), open=False):
        with gr.Row():
            c["f0_autotune"] = gr.Checkbox(value=defaults.f0_autotune, label=i18n("Autotune"),
                                           info=i18n("Snap the pitch to the nearest note"))
            c["f0_autotune_strength"] = gr.Slider(0, 1, value=defaults.f0_autotune_strength, step=0.05,
                                                  label=i18n("Autotune strength"))
        with gr.Row():
            c["proposed_pitch"] = gr.Checkbox(value=defaults.proposed_pitch, label=i18n("Automatic pitch"),
                                              info=i18n("Move the singing to the range of the voice model"))
            c["proposed_pitch_threshold"] = gr.Slider(50, 1200, value=defaults.proposed_pitch_threshold, step=1,
                                                      label=i18n("Voice model range (Hz)"),
                                                      info=i18n("About 155 for male models and 255 for female models"))
        with gr.Row():
            c["embedder_model"] = gr.Dropdown(EMBEDDERS, value=defaults.embedder_model, label=i18n("Embedder model"),
                                              info=i18n("Use the one the voice model was trained with (usually contentvec)"))
            c["embedder_model_custom"] = gr.Textbox(value=defaults.embedder_model_custom or "", lines=1, max_lines=1,
                                                    label=i18n("Custom embedder folder"),
                                                    placeholder=i18n("Folder with pytorch_model.bin and config.json"))
        with gr.Row():
            c["formant_shifting"] = gr.Checkbox(value=defaults.formant_shifting, label=i18n("Formant shifting"),
                                                info=i18n("Change the timbre, e.g. for male to female conversions"))
            c["formant_qfrency"] = gr.Slider(0, 16, value=defaults.formant_qfrency, step=0.1, label=i18n("Quefrency"))
            c["formant_timbre"] = gr.Slider(0, 16, value=defaults.formant_timbre, step=0.1, label=i18n("Timbre"))
        with gr.Row():
            c["cleanup_mode"] = gr.Dropdown(CLEANUP_MODES, value=defaults.cleanup_mode, label=i18n("Noise cleanup"),
                                            info=i18n("speech may remove parts of the singing"))
            c["cleanup_strength"] = gr.Slider(0, 30, value=defaults.cleanup_strength, step=1,
                                              label=i18n("Cleanup strength"))
    return c


def rvc_params(values: dict) -> RvcParams:
    """``RvcParams`` from handler kwargs; unrelated keys are ignored."""
    kwargs = {k: v for k, v in values.items() if k in RVC_FIELDS}
    if "pitch" in kwargs:
        kwargs["pitch"] = int(kwargs["pitch"])
    kwargs["embedder_model_custom"] = (kwargs.get("embedder_model_custom") or "").strip() or None
    return RvcParams(**kwargs)
