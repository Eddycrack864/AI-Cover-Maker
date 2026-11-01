import logging

import gradio as gr

from ai_cover_maker.core.device import get_device
from ai_cover_maker.core.pipeline.cache import clear_workspace, workspace_size
from ai_cover_maker.core.pipeline.options import load_presets
from ai_cover_maker.i18n import AUTO_LANGUAGE, LANGUAGE_NAMES, i18n, normalize_language
from ai_cover_maker.ui import a11y
from ai_cover_maker.ui.adapters import user_errors
from ai_cover_maker.ui.context import AppContext
from ai_cover_maker.ui.tabs.cover import CoverTab, default_settings

logger = logging.getLogger(__name__)


def render_settings_tab(ctx: AppContext, cover: CoverTab) -> None:
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

        with gr.Accordion(i18n("Save settings"), open=False):
            gr.Markdown(i18n("Save the current settings of the Cover tab to load them every time the app starts, "
                             "or go back to the default settings"))
            with gr.Row():
                save_settings_button = gr.Button(i18n("Save current settings"), variant="primary")
                reset_settings_button = gr.Button(i18n("Reset settings to default"), variant="primary")

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

    # Cover settings
    @user_errors
    def save_settings(*values):
        ctx.cover_settings.save(dict(zip(cover.settings, values)))
        gr.Info(i18n("Settings saved. They will be loaded every time the app starts"))

    @user_errors
    def reset_settings():
        ctx.cover_settings.reset()
        defaults = default_settings(load_presets())  # the first voice may have changed since the app started
        gr.Info(i18n("Settings reset to default"))
        return [defaults[name] for name in cover.settings]

    save_settings_button.click(save_settings, inputs=list(cover.settings.values()), api_visibility="undocumented")
    reset_settings_button.click(reset_settings, outputs=list(cover.settings.values()),
                                api_visibility="undocumented")

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

