"""Assembles the Gradio Blocks app from the individual tabs."""

import gradio as gr

from ai_cover_maker import paths
from ai_cover_maker.i18n import i18n
from ai_cover_maker.ui import a11y, theme
from ai_cover_maker.ui.context import AppContext
from ai_cover_maker.ui.tabs.cover import refresh_cover_voices, render_cover_tab
from ai_cover_maker.ui.tabs.credits import render_credits_tab
from ai_cover_maker.ui.tabs.settings import render_settings_tab
from ai_cover_maker.ui.tabs.themes import render_themes_tab
from ai_cover_maker.ui.tabs.voices import render_voices_tab

TITLE = "🎤 AI Cover Maker 🎤"
# Every uploaded audio and every returned stem is copied to Gradio's temp cache: once a
# day, delete the files older than a day so it doesn't grow without limit.
DELETE_CACHE = (86400, 86400)


def launch_kwargs(ctx: AppContext) -> dict:
    """App-level options that Gradio 6 takes in ``launch()`` instead of ``gr.Blocks()``."""
    return {
        "theme": theme.load_theme(ctx.config) or theme.DEFAULT_THEME,
        "js": theme.mode_js(theme.current_mode(ctx.config)),
        "head": a11y.head_html(i18n.language),
        "css": a11y.CSS,
        "i18n": a11y.gradio_i18n(i18n.language),
        # On Colab outputs/ can be a symlink to Google Drive. Gradio resolves symlinks before
        # checking a returned file, so the cwd alone doesn't cover it.
        "allowed_paths": [str(paths.OUTPUTS_DIR)],
    }


def create_app(ctx: AppContext = None) -> gr.Blocks:
    ctx = ctx or AppContext.create()

    # No Gradio telemetry: it reports every launch to huggingface.co
    with gr.Blocks(title=TITLE, analytics_enabled=False, delete_cache=DELETE_CACHE) as app:
        # The emojis are hidden from screen readers. gr.HTML, because the Markdown sanitizer drops aria-hidden
        gr.HTML('<h1><span aria-hidden="true">🎤</span> AI Cover Maker <span aria-hidden="true">🎤</span></h1>')
        gr.Markdown(i18n("If you like AI Cover Maker you can star my repo on [GitHub](https://github.com/Eddycrack864/AI-Cover-Maker)"))

        with gr.Tabs():
            cover = render_cover_tab(ctx)
            selectors = [cover.voice, cover.backing_voice]
            render_voices_tab((refresh_cover_voices, selectors, selectors))
            render_themes_tab(ctx)
            render_settings_tab(ctx, cover)
            render_credits_tab()

    return app
