import gradio as gr

from ai_cover_maker.i18n import i18n
from ai_cover_maker.ui import a11y

CREDITS = """
AI Cover Maker maintained by **[Eddycrack 864](https://github.com/Eddycrack864).** Join **[AI HUB](https://discord.gg/aihub)** community.
* [Applio](https://github.com/IAHispano/Applio) by [IAHispano](https://github.com/IAHispano): RVC inference (MIT).
* [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) by [beveradb](https://github.com/beveradb).
* [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by [Anjok07](https://github.com/Anjok07).
* [UVR5 UI](https://github.com/Eddycrack864/UVR5-UI), the base of this interface.
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [Pedalboard](https://github.com/spotify/pedalboard) devs.
* [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) by [SociallyIneptWeeb](https://github.com/SociallyIneptWeeb), the original idea.
* [ShiromiyaG](https://github.com/ShiromiyaG), original creator of RVC AI Cover Maker UI.
* [Nick088](https://linktr.ee/Nick088) and [FullmatheusBallZ](https://www.youtube.com/@FullmatheusBallZ) for helping with the original Colab notebooks.
* Every author of the separation models and voice models you use.
"""


def render_credits_tab() -> None:
    with gr.TabItem(i18n("Credits")):
        a11y.tab_heading(i18n("Credits"))
        gr.Markdown(CREDITS)
