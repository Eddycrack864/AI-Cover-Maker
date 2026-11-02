<h1 align="center"><b> 🎤 AI Cover Maker 🎤 </b></h1>
<div align="center">

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/Eddycrack864/AI-Cover-Maker)

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Community-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/aihub)

Make AI covers in one click: the song is separated with [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), the vocals are converted with RVC ([Applio](https://github.com/IAHispano/Applio)) and everything is mixed back together.
The rewrite of [RVC AI Cover Maker UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI), built on [UVR5 UI](https://github.com/Eddycrack864/UVR5-UI). Originally created for the [AI HUB](https://discord.gg/aihub) community.

</div>

<div align="center">

**[Docs](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/info/docs.md) / [Troubleshooting](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/info/troubleshooting.md)**

</div>

## Features:
* User friendly interface: a song (file or link) in, a cover out.
* Separation with the best UVR models (Roformer, MDX23C, MDX-NET, VR Arch): vocals/instrumental, lead/backing vocals, de-reverb, de-echo and denoise. Each step only lists the models made for it, with a recommended model and three presets (Fast, Balanced, Best quality).
* Voice conversion with RVC (Applio): rmvpe/fcpe/crepe/swift, index, protect, autotune, automatic pitch, formant shifting, custom embedders, noise cleanup.
* Convert the backing vocals with a second voice, or keep them.
* Change the key of the whole song keeping its tempo.
* Vocal effects (high-pass, compressor, reverb) and a mix that never clips.
* Every step is cached: trying another voice, effect or mix on the same song only recomputes what changed.
* Each cover is saved in its own folder with every stem made along the way, for your own remixes.
* Add voice models by link (Hugging Face, Google Drive, direct links) or upload them.
* Songs from all sites supported by [yt_dlp](https://github.com/yt-dlp/yt-dlp). Check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).
* Batch covers.
* Save your settings, themes and 19 languages.
* Colab support.
* Windows/Linux support.

## Requirements

### Hardware Requirements:
* NVIDIA GTX 10 series or newer (CUDA 12.8).
* About 15 GB of disk space.

> [!NOTE]
> CPU works, but it will be very slow. If you don't meet the hardware requirements use our [Colab](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb).

### Prerequisites:
- Git. You can download Git [here](https://git-scm.com/downloads).
- FFmpeg. You can download FFmpeg [here](https://www.ffmpeg.org/download.html) or you can use my [automated installation script](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/info/ffmpeg-installer.bat) (for Windows).
- For linux users, run this command on a terminal: (Debian and Ubuntu users): `sudo apt install ffmpeg git build-essential` (For Arch linux users): `sudo pacman -S ffmpeg git base-devel` (For Fedora users): `sudo dnf install ffmpeg git gcc gcc-c++`
(Some distributions already come with Git, FFmpeg and a C compiler preinstalled so this step may be optional. The compiler is needed to build one of the dependencies during installation.)

> [!IMPORTANT]
> FFmpeg has to be added to the PATH. (only needed on Windows)

## Getting Started

Clone the repository (git needed) or download the source code of the latest release [here](https://github.com/Eddycrack864/AI-Cover-Maker/releases)

```
git clone https://github.com/Eddycrack864/AI-Cover-Maker.git
```

Then continue with the steps described below

### 1. Installation

Run the installation script based on your operating system:

- **Windows:** Double-click `AI-Cover-Maker-installer.bat` (DONT RUN AS ADMINISTRATOR 🚧).
- **Linux:** Run `AI-Cover-Maker-installer.sh` with `chmod +x AI-Cover-Maker-installer.sh` and `./AI-Cover-Maker-installer.sh`.

It creates a private Python 3.12 environment in the `env/` folder, so your system Python is not touched.

### 2. Running AI Cover Maker

Start AI Cover Maker using:

- **Windows:** Double-click `run-AI-Cover-Maker.bat`.
- **Linux:** Run `run-AI-Cover-Maker.sh` with `chmod +x run-AI-Cover-Maker.sh` and `./run-AI-Cover-Maker.sh`.

Extra options can be added to the run scripts (or to `python app.py`):

| Option | Description |
|---|---|
| `--share` | Create a temporary public Gradio link |
| `--listen-port 8080` | Use another port (default: 7755) |
| `--server-name 0.0.0.0` | Make the UI reachable from other devices on your network |
| `--open` | Open the UI in your browser (the run scripts already do this) |

Example: `run-AI-Cover-Maker.bat --share --listen-port 8080`

### 3. Update AI Cover Maker (If you want/need it)

Update AI Cover Maker using (git needed):

- **Windows:** Double-click `AI-Cover-Maker-updater.bat`.
- **Linux:** Run `AI-Cover-Maker-updater.sh` with `chmod +x AI-Cover-Maker-updater.sh` and `./AI-Cover-Maker-updater.sh`.

Your settings (`user_data/`), models (`models/`) and covers (`outputs/`) are kept; updating never touches them.

If you find an error when installing or running the program please consult [this troubleshooting file](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/info/troubleshooting.md) first, if your error is not described there please create an [issue](https://github.com/Eddycrack864/AI-Cover-Maker/issues)

### 4. Debug (If you want/need it)

Check what AI Cover Maker runs on (GPU, CUDA, FFmpeg):

- **Windows:** Double-click `info/status-checker.bat`.
- **Linux:** Run `info/status-checker.sh` with `chmod +x info/status-checker.sh` and `./info/status-checker.sh`.

## How to use

1. Add a voice model in the **Voices** tab (a link or the `.pth` and `.index` files), then select it in the **Cover** tab.
2. Drop a song, or paste a link in **Download a song by link**.
3. Set the **Pitch** if needed (see below) and press **Make cover!**.

Your cover is saved in `outputs/<song> (<voice> Cover)/`, together with every stem made along the way.

### Separation steps
Choose a **preset** (Fast, Balanced or Best quality) or the model of each step. Each list only shows the models made for that step, the recommended one marked with a ⭐ (**Show all models** lists them all).

| Step | What it does |
|---|---|
| **Vocals / instrumental model** | Separates the vocals from the instrumental. Required |
| **Lead / backing vocals model** | Optional. Converts only the lead vocals and keeps the backing vocals apart |
| **De-reverb model** | Optional. Removes the reverb: RVC sounds much cleaner on dry vocals |
| **De-echo model** | Optional. Removes echo/delay from the lead vocals |
| **Denoise model** | Optional. Removes noise from the lead vocals |

### Voice conversion settings (RVC)
| Setting | What it does |
|---|---|
| **Pitch** | Semitones. +12 for a male voice with a female model, -12 for the opposite |
| **Pitch extraction method** | How the melody is detected. **rmvpe** is the best for singing |
| **Search feature ratio** | How much the `.index` file is used: more accent of the model, but more artifacts |
| **Protect voiceless consonants** | Keeps breaths and consonants clean. 0.5 disables it |
| **Volume envelope** | 1 keeps the loudness of the converted voice, 0 follows the original singer |
| **More conversion settings** | Autotune, automatic pitch, embedder model, formant shifting and noise cleanup |

### Vocal effects and mix
| Setting | What it does |
|---|---|
| **High-pass filter** | Removes rumble below the cutoff frequency |
| **Compressor** | Evens out the vocal volume |
| **Reverb** | Puts the dry converted vocals back in a room |
| **Change the key of the song** | Semitones for the whole song (instrumental and vocals), keeping the tempo |
| **Lead / backing vocals / instrumental volume** | The volume of each part in the cover, in dB |

Every setting is explained in detail in the [docs](https://github.com/Eddycrack864/AI-Cover-Maker/blob/main/info/docs.md). You can save your settings in **Settings > Save settings** so they are loaded every time AI Cover Maker starts.

## Precompiled Version
The precompiled version already includes the Python environment, so there is nothing to install: download, extract and run.

1. Get the precompiled version for your PC (the latest one is also linked in each [release](https://github.com/Eddycrack864/AI-Cover-Maker/releases/latest)):
   - **[Windows](https://huggingface.co/Eddycrack864/AI-Cover-Maker/tree/main/Windows)** (`.zip`)
   - **[Linux](https://huggingface.co/Eddycrack864/AI-Cover-Maker/tree/main/Linux)** (`.tar.gz`, needs glibc 2.35 or newer: Ubuntu 22.04, Debian 12, Fedora 36 or later)

2. Extract it, it contains an `AI-Cover-Maker` folder:
   - **Windows:** right click > "Extract All..." (or "Extract here" with 7-Zip/WinRAR).
   - **Linux:** `tar -xzf AI-Cover-Maker-vX.X.X-linux.tar.gz`
3. Run `run-AI-Cover-Maker.bat` (Windows) or `./run-AI-Cover-Maker.sh` (Linux). You can now use all the features of the normal installation.

FFmpeg is still needed (see the [prerequisites](#prerequisites)).

> [!NOTE]
> Still, to update AI Cover Maker you need to install Git.

## Development

The app is a Python package, `ai_cover_maker/`, organised in layers:

| Path | What lives there |
|---|---|
| `app.py` | Thin launcher (`python app.py [options]`) |
| `ai_cover_maker/core/` | Separation, RVC (`rvc/`), effects, mixer, and the cover `pipeline/` with its stage cache. No Gradio imports, testable on its own |
| `ai_cover_maker/ui/` | Gradio layer, one module per tab in `ui/tabs/` |
| `third_party/applio/` | Inference subset of Applio, vendored by `tools/sync_applio.py` (never edit it by hand) |
| `assets/` | Data only: model catalog (`models.json`), pipeline presets, default settings, translations, themes |
| `info/` | Docs, troubleshooting, the FFmpeg installer and the status checkers |

Common tasks:

```bash
# Install the test tools (once) and run the tests
env/python.exe -m pip install -r requirements-dev.txt   # Linux: env/bin/python
env/python.exe -m pytest                                 # --slow: real models, --network: online checks

# Check that every translatable string exists in en_US.json and in every other language
env/python.exe tools/i18n_scan.py --check --langs

# Model catalog (needs internet)
env/python.exe tools/sync_models.py --write              # add the models audio-separator supports
env/python.exe tools/tag_model_stages.py --write         # tag each model with the separation steps it fits

# Re-vendor Applio from a checkout in .reference/Applio
env/python.exe tools/sync_applio.py
```

- **Translations:** the English text passed to `i18n("...")` is the key in every language file, so changing it breaks existing translations. A new string must be added to all the files in `assets/i18n/languages/`; `tests/test_i18n_scan.py` fails if a language is missing one.

## Contributions
If you want to participate and help me with this project feel free to create an [issue](https://github.com/Eddycrack864/AI-Cover-Maker/issues) if something goes wrong or make a [pull request](https://github.com/Eddycrack864/AI-Cover-Maker/pulls) to improve this project.

Any type of contribution is welcome 💖

If you like this project you can star this repository. I will appreciate a lot 💖💖💖

## Credits
* AI Cover Maker maintained by [Eddycrack 864](https://github.com/Eddycrack864).
* [Applio](https://github.com/IAHispano/Applio) by [IAHispano](https://github.com/IAHispano) for the RVC inference (MIT).
* python-audio-separator by [beveradb](https://github.com/beveradb).
* [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by [Anjok07](https://github.com/Anjok07).
* [UVR5 UI](https://github.com/Eddycrack864/UVR5-UI), the base of this interface.
* Thanks to [yt_dlp](https://github.com/yt-dlp/yt-dlp) and [Pedalboard](https://github.com/spotify/pedalboard) devs.
* [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) by [SociallyIneptWeeb](https://github.com/SociallyIneptWeeb), the original idea.
* [ShiromiyaG](https://github.com/ShiromiyaG), original creator of RVC AI Cover Maker UI.
* Thanks to [Nick088](https://linktr.ee/Nick088) and [FullmatheusBallZ](https://www.youtube.com/@FullmatheusBallZ) for helping with the original Colab notebooks.
* Every author of the separation models and voice models you use.

> [!WARNING]
> Only use voices and songs you have the rights to.
