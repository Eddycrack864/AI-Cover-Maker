<h1 align="center"><b> 🎤 AI Cover Maker 🎤 </b></h1>
<div align="center">

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Community-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/aihub)

Make AI covers in one click: the song is separated with [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), the vocals are converted with RVC ([Applio](https://github.com/IAHispano/Applio)) and everything is mixed back together.
The rewrite of [RVC AI Cover Maker UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI), built on [UVR5 UI](https://github.com/Eddycrack864/UVR5-UI). Originally created for the [AI HUB](https://discord.gg/aihub) community.

</div>

## Features
* **Cover**: song (file or link) → vocals/instrumental separation → optional lead/backing split, de-reverb, de-echo and denoise → RVC → vocal effects → mix. Presets (Fast, Balanced, Best quality) or pick every model yourself.
* Every step is cached: trying another mix, effect, key or voice on the same song only recomputes what changed.
* Convert the backing vocals with a second voice, or keep them.
* Change the key of the whole song keeping its tempo.
* Vocal effects: high-pass, compressor and reverb. A limiter keeps the cover from clipping.
* Every Applio conversion option: rmvpe/fcpe/crepe/swift, index ratio, protect, autotune, automatic pitch, formant shifting, custom embedders, noise cleanup.
* Each cover is saved in its own folder with every stem made along the way (vocals, instrumental, lead/backing vocals, reverb, converted vocals...), named after the song, for your own remixes.
* **Voices**: add voice models by link (Hugging Face, Google Drive, direct `.zip`/`.pth`/`.index`) or upload, and delete them.
* The full UVR5 UI model catalog (Roformer, MDX23C, MDX-NET, VR Arch, Demucs) for every separation step, with UVR5 UI's default parameters.
* Batch covers, downloads from [every site yt-dlp supports](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md), themes, English/Spanish/Portuguese.
* Windows/Linux and Google Colab.

## Requirements

* NVIDIA GPU recommended (CUDA 12.8 build of PyTorch, GTX 10 series or newer). It works on CPU, just slowly; if you don't have a GPU use the [Colab](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb).
* About 15 GB of disk space for the environment and models.
* [Git](https://git-scm.com/downloads) and [FFmpeg](https://www.ffmpeg.org/download.html) on the PATH. On Windows you can use `tools/ffmpeg-installer.bat`. On Linux: `sudo apt install ffmpeg git build-essential` (Debian/Ubuntu).

## Getting started

```
git clone https://github.com/Eddycrack864/AI-Cover-Maker.git
```

1. **Install**: double-click `AI-Cover-Maker-installer.bat` (Windows, not as Administrator) or run `./AI-Cover-Maker-installer.sh` (Linux). It creates a private Python 3.12 environment in `env/`, so your system Python is not touched.
2. **Run**: `run-AI-Cover-Maker.bat` or `./run-AI-Cover-Maker.sh`.
3. **Update**: `AI-Cover-Maker-updater.bat` or `./AI-Cover-Maker-updater.sh`. Your settings (`user_data/`), models (`models/`) and covers (`outputs/`) are kept.

Run options (also for `python app.py`):

| Option | Description |
|---|---|
| `--share` | Create a temporary public Gradio link |
| `--listen-port 8080` | Use another port (default: 7755) |
| `--server-name 0.0.0.0` | Make the UI reachable from other devices on your network |
| `--open` | Open the UI in your browser (the run scripts already do this) |

Voice models live in `models/rvc/<name>/` (`.pth` plus an optional `.index`). Covers are saved in `outputs/<song> (<voice> Cover)/`, cached steps in `workspace/` (clear it from Settings).

## Development

| Path | What lives there |
|---|---|
| `ai_cover_maker/core/` | No Gradio: separation, RVC (`rvc/`), effects, mixer, and the cover `pipeline/` with its stage cache |
| `ai_cover_maker/ui/` | Gradio layer, one module per tab in `ui/tabs/` |
| `third_party/applio/` | Inference subset of Applio, vendored by `tools/sync_applio.py` (never edit it by hand) |
| `assets/` | Model catalog, pipeline presets, default settings, translations, themes |

```bash
env/python.exe -m pip install -r requirements-dev.txt   # Linux: env/bin/python
env/python.exe -m pytest                                 # --slow: real models, --network: online checks
env/python.exe tools/i18n_scan.py --check --langs
env/python.exe tools/sync_applio.py                      # re-vendor Applio from a checkout in .reference/Applio
```

## Credits
See the Credits tab. AI Cover Maker uses [Applio](https://github.com/IAHispano/Applio) (MIT, © AI Hispano) for RVC inference and [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) for separation; voice and separation models belong to their authors.

Only use voices and songs you have the rights to.
