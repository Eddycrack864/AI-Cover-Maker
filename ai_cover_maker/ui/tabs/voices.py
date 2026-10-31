"""The Voices tab: add (link or upload) and remove RVC voice models."""

from pathlib import Path

import gradio as gr

from ai_cover_maker.core.rvc.voices import IMPORT_EXTENSIONS, delete_voice, download_voice, install_voice, list_voices
from ai_cover_maker.i18n import N_, i18n
from ai_cover_maker.ui import a11y, components
from ai_cover_maker.ui.adapters import require, user_errors
from ai_cover_maker.ui.rvc_controls import voice_names


def voices_table() -> list[list[str]]:
    return [[v.name, v.model_path.name, v.index_path.name if v.index_path else "—"] for v in list_voices()]


def render_voices_tab(on_change) -> None:
    """``on_change`` is ``(fn, inputs, outputs)`` refreshing the voice selectors of the other tabs."""
    with gr.TabItem(i18n("Voices")):
        a11y.tab_heading(i18n("Voices"))
        table = gr.Dataframe(value=voices_table(), headers=[i18n("Voice model"), i18n("Model file"), i18n("Index file")],
                             label=i18n("Installed voice models"), interactive=False, wrap=True)

        with gr.Accordion(i18n("Download a voice model by link"), open=True):
            gr.Markdown(i18n("Links to Hugging Face, Google Drive or a direct link to a .zip, .pth or .index file"))
            with gr.Row():
                link = components.single_line_textbox(i18n("Link"), i18n("Paste the link here"))
                link_name = components.single_line_textbox(i18n("Name"), i18n("Optional, the file name by default"))
            download = gr.Button(i18n("Download voice model"), variant="primary")

        with gr.Accordion(i18n("Upload a voice model"), open=False):
            files = gr.File(label=i18n("Model files (.pth, .index or .zip)"), file_count="multiple",
                            file_types=list(IMPORT_EXTENSIONS))
            upload_name = components.single_line_textbox(i18n("Name"), i18n("Name of the voice model"))
            upload = gr.Button(i18n("Add voice model"), variant="primary")

        with gr.Accordion(i18n("Delete a voice model"), open=False):
            with gr.Row():
                to_delete = gr.Dropdown(choices=voice_names(), label=i18n("Voice model to delete"), interactive=True)
                delete = gr.Button(i18n("Delete voice model"), variant="stop", scale=0)
        status = gr.Markdown()

    def changed(message: str):
        return voices_table(), gr.update(choices=voice_names(), value=None), message

    @user_errors
    def run_download(url, name):
        folder = download_voice(require(url, N_("No link provided")), name)
        return changed(i18n("Voice model added: {name}").format(name=folder.name))

    @user_errors
    def run_upload(paths, name):
        folder = install_voice([Path(p) for p in (paths or [])], require(name, N_("Please write a name for the voice model")))
        return changed(i18n("Voice model added: {name}").format(name=folder.name))

    @user_errors
    def run_delete(name):
        delete_voice(require(name, N_("Please select a voice model")))
        return changed(i18n("Voice model deleted: {name}").format(name=name))

    outputs = [table, to_delete, status]
    refresh_fn, refresh_inputs, refresh_outputs = on_change
    for button, fn, inputs, api_name in ((download, run_download, [link, link_name], "download_voice"),
                                         (upload, run_upload, [files, upload_name], "upload_voice"),
                                         (delete, run_delete, [to_delete], "delete_voice")):
        button.click(fn, inputs=inputs, outputs=outputs, api_name=api_name).success(
            refresh_fn, inputs=refresh_inputs, outputs=refresh_outputs, api_visibility="undocumented")
