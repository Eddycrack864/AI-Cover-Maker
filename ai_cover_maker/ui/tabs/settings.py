import logging

import gradio as gr

from ai_cover_maker.core.device import get_device
from ai_cover_maker.core.pipeline.cache import clear_workspace, workspace_size
from ai_cover_maker.i18n import AUTO_LANGUAGE, LANGUAGE_NAMES, i18n, normalize_language
from ai_cover_maker.ui import a11y
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

        with gr.Accordion(i18n("Cover cache"), open=False):
            gr.Markdown(i18n("Every step of a cover is saved in the workspace folder, so trying other mix or voice "
                             "settings on the same song is fast. Clear it to free disk space."))
            with gr.Row():
                cache_size = gr.Markdown(_cache_size_text())
                clear_cache = gr.Button(i18n("Clear the cache"), variant="primary", scale=0)

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

    # Cache
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

