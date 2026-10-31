"""Application entry point: parse options, prepare user data and launch Gradio."""

import logging

from ai_cover_maker import paths
from ai_cover_maker.cli import parse_args
from ai_cover_maker.config.store import migrate_legacy_user_data


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "gradio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    migrate_legacy_user_data()

    from ai_cover_maker.ui.app import create_app, launch_kwargs
    from ai_cover_maker.ui.context import AppContext

    ctx = AppContext.create()
    create_app(ctx).queue().launch(
        share=options.share,
        favicon_path=str(paths.FAVICON),
        server_name=options.server_name,
        server_port=options.listen_port,
        inbrowser=options.open_browser,
        run_history=options.run_history,
        **launch_kwargs(ctx),
    )
