import logging

import gradio as gr

from ai_cover_maker.core.device import get_device
from ai_cover_maker.core.downloads.model_downloader import METHOD_PYTHON, METHODS, SOURCES, download_model
from ai_cover_maker.core.pipeline.cache import clear_workspace, workspace_size
from ai_cover_maker.core.separation.runtime import RuntimeOptions
from ai_cover_maker.i18n import AUTO_LANGUAGE, LANGUAGE_NAMES, N_, i18n, normalize_language
from ai_cover_maker.ui import a11y
from ai_cover_maker.ui.adapters import progress_callback, require, user_errors
from ai_cover_maker.ui.components import model_choices, output_log
from ai_cover_maker.ui.context import AppContext

logger = logging.getLogger(__name__)


def render_settings_tab(ctx: AppContext) -> None:
    with gr.TabItem(i18n("Settings")):
        a11y.tab_heading(i18n("Settings"))
        with gr.Accordion(i18n("Language selector"), open=False):
            selected_language = gr.Dropdown(
                label=i18n("Language"),
                info=i18n("Select the language you want to use. (Requires restarting the App)"),
                value=_current_language(ctx),
                # Names instead of bare codes, which screen readers spell out ("a r underscore A R")
                choices=[(i18n("Language automatically detected by system"), AUTO_LANGUAGE)]
                + [(f"{LANGUAGE_NAMES.get(code, code)} ({code})", code) for code in i18n.available_languages()],
                interactive=True,
            )

        with gr.Accordion(i18n("Alternative model downloader"), open=False):
            download_method = gr.Dropdown(
                label=i18n("Download method"),
                info=i18n("Select the download method you want to use. (Must have it installed)"),
                value=METHOD_PYTHON,
                choices=METHODS,
                interactive=True,
            )
            download_source = gr.Dropdown(
                label=i18n("Download source"),
                info=i18n("Select the source to download the model from (Github or Hugging Face)"),
                value=lambda: None,
                choices=list(SOURCES),
                interactive=True,
            )
            model_to_download = gr.Dropdown(
                label=i18n("Model to download"),
                info=i18n("Select the model to download using the selected method"),
                value=lambda: None,
                choices=model_choices(ctx.catalog, ctx.catalog.downloadable()),
                interactive=True,
            )
            download_button = gr.Button(i18n("Download!"), variant="primary")
            download_output = output_log()

        with gr.Accordion(i18n("Performance"), open=False):
            runtime = RuntimeOptions.from_config(ctx.config)
            precision = gr.Dropdown(
                label=i18n("Precision"),
                info=i18n("Native FP16 makes BS/Mel Roformer models faster and lighter on NVIDIA and Apple GPUs. "
                          "Unsupported combinations fall back to FP32."),
                choices=[(i18n("Automatic (autocast on CUDA)"), "auto"), ("FP32", "fp32"), ("Autocast", "autocast"),
                         (i18n("Native FP16"), "native_fp16")],
                value=runtime.precision,
                interactive=True,
            )
            torch_compile = gr.Checkbox(
                label=i18n("Compile BS/Mel Roformer models (torch.compile)"),
                info=i18n("Experimental: speeds up long or repeated separations, but the first one is slower. "
                          "Needs Triton (Linux, Colab); without it, it is skipped."),
                value=runtime.torch_compile,
                interactive=True,
            )
            chunk_duration = gr.Number(
                label=i18n("Process long audio in chunks (seconds)"),
                info=i18n("0 = disabled. Use 600 for audio longer than an hour to avoid running out of memory. "
                          "Chunks are joined without a crossfade."),
                value=runtime.chunk_duration,
                minimum=0,
                step=60,
                precision=0,
                interactive=True,
            )
            use_soundfile = gr.Checkbox(
                label=i18n("Write outputs with soundfile"),
                info=i18n("Uses less memory when writing very long files"),
                value=runtime.use_soundfile,
                interactive=True,
            )
            gr.Markdown(i18n("Changes apply to the next cover."))

        with gr.Accordion(i18n("Cover cache"), open=False):
            gr.Markdown(i18n("Every step of a cover is saved in the workspace folder, so trying other mix or voice "
                             "settings on the same song is fast. Clear it to free disk space."))
            with gr.Row():
                cache_size = gr.Markdown(_cache_size_text())
                clear_cache = gr.Button(i18n("Clear the cache"), variant="stop", scale=0)

        with gr.Accordion(i18n("Discord Rich Presence"), open=False):
            discord_presence = gr.Checkbox(
                label=i18n("Show AI Cover Maker activity on Discord (Requires restarting the App)"),
                value=bool(ctx.config.get("discord_presence", False)),
                interactive=True,
            )

        gr.Markdown(f"{i18n('Device')}: `{get_device()}`")

    # Language
    def save_language(language):
        if language == AUTO_LANGUAGE:
            ctx.config.set("lang.override", False)
        else:
            ctx.config.set("lang.override", True)
            ctx.config.set("lang.selected_lang", language)
        gr.Info(i18n("Language has been saved. Restart AI Cover Maker to apply the changes"))

    selected_language.change(save_language, inputs=[selected_language], api_visibility="undocumented")

    # Alternative model downloader
    @user_errors
    def run_download(method, source, model_id, progress=gr.Progress()):
        require(model_id, N_("Please select a model first"))
        model = ctx.catalog.find(model_id)
        lines = []
        for line in download_model(model, source, method or METHOD_PYTHON, progress=progress_callback(progress)):
            lines.append(line)
            yield "\n".join(lines)

    download_button.click(run_download, inputs=[download_method, download_source, model_to_download],
                          outputs=[download_output], api_name="download_model")

    # Performance
    for component, key, convert in (
        (precision, "performance.precision", str),
        (torch_compile, "performance.torch_compile", bool),
        (chunk_duration, "performance.chunk_duration", lambda v: max(int(v or 0), 0)),
        (use_soundfile, "performance.use_soundfile", bool),
    ):
        component.change(lambda value, key=key, convert=convert: ctx.config.set(key, convert(value)),
                         inputs=[component], api_visibility="undocumented")

    def run_clear_cache():
        removed = clear_workspace()
        gr.Info(i18n("Cache cleared ({count} songs)").format(count=removed))
        return _cache_size_text()

    clear_cache.click(run_clear_cache, outputs=[cache_size], api_visibility="undocumented")

    # Discord
    def save_discord(enabled):
        ctx.config.set("discord_presence", bool(enabled))

    discord_presence.change(save_discord, inputs=[discord_presence], api_visibility="undocumented")


def _cache_size_text() -> str:
    return i18n("Cache size: {size} MB").format(size=round(workspace_size() / 2**20, 1))


def _current_language(ctx: AppContext) -> str:
    language = normalize_language(ctx.config.get("lang.selected_lang"))
    if not ctx.config.get("lang.override", False) or language not in i18n.available_languages():
        return AUTO_LANGUAGE
    return language

