# AI Cover Maker Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running AI Cover Maker](#running-ai-cover-maker)
5. [Updating AI Cover Maker](#updating-ai-cover-maker)
6. [How a cover is made](#how-a-cover-is-made)
7. [Separation steps](#separation-steps)
8. [Voice conversion settings (RVC)](#voice-conversion-settings-rvc)
9. [Backing vocals](#backing-vocals)
10. [Vocal effects](#vocal-effects)
11. [Mix](#mix)
12. [Batch covers](#batch-covers)
13. [Output files and the cover cache](#output-files-and-the-cover-cache)
14. [Voices tab](#voices-tab)
15. [Settings tab](#settings-tab)
16. [Folders](#folders)
17. [Tips for better covers](#tips-for-better-covers)
18. [Contributions](#contributions)
19. [Credits](#credits)
20. [Feedback and Support](#feedback-and-support)

## Introduction
AI Cover Maker makes an AI cover of a song in one click. It separates the song with
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (the models of Ultimate Vocal Remover),
converts the vocals with RVC (the inference code of [Applio](https://github.com/IAHispano/Applio)), adds vocal effects and
mixes everything back together. It is a from-scratch rewrite of
[RVC AI Cover Maker UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI), built on [UVR5 UI](https://github.com/Eddycrack864/UVR5-UI).

## Requirements
### Hardware
- **NVIDIA GPU** recommended: GTX 10 series or newer. AI Cover Maker uses the CUDA 12.8 build of PyTorch, so keep
  your NVIDIA driver up to date.
- It also works on the **CPU**, but a cover takes much longer. If you don't have an NVIDIA GPU, use the
  [Colab](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb).
- **Disk space**: about 15 GB for the Python environment and the models, plus the space of your covers.
- GPUs with 8 GB of VRAM or less work too: the separation model and the voice model are never loaded at the same time.

### Software
- **Git**, to install and update: [download it here](https://git-scm.com/downloads).
- **FFmpeg** on the PATH:
  - **Windows**: run [`info/ffmpeg-installer.bat`](ffmpeg-installer.bat). It installs FFmpeg for your user (no
    Administrator rights needed) and adds it to the PATH. Alternatively: `winget install Gyan.FFmpeg`.
  - **Linux**: install it with your package manager, together with Git and a C compiler (one dependency is built
    during the installation):
    ```bash
    sudo apt install ffmpeg git build-essential     # Debian/Ubuntu
    sudo pacman -S ffmpeg git base-devel            # Arch Linux
    sudo dnf install ffmpeg git gcc gcc-c++         # Fedora
    ```

## Installation
### Normal installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Eddycrack864/AI-Cover-Maker.git
   ```
2. Run the installer:
   - **Windows**: double-click `AI-Cover-Maker-installer.bat` (do not run it as Administrator).
   - **Linux**: `chmod +x AI-Cover-Maker-installer.sh && ./AI-Cover-Maker-installer.sh` (do not use sudo).

The installer installs Miniconda in your user folder if you don't have it, and creates a private Python 3.12
environment in the `env/` folder of AI Cover Maker with PyTorch 2.7.1 (CUDA 12.8) and the rest of the dependencies.
Your system Python is not touched. It downloads about 4 GB.

> **Tip**: install AI Cover Maker in a simple path without spaces or special characters, e.g. `C:\AI-Cover-Maker`.

### Precompiled version
The precompiled version already includes the Python environment: download, extract and run.

1. Download it:
   - **[Windows](https://huggingface.co/Eddycrack864/AI-Cover-Maker/tree/main/Windows)** (`.zip`)
   - **[Linux](https://huggingface.co/Eddycrack864/AI-Cover-Maker/tree/main/Linux)** (`.tar.gz`, needs glibc 2.35 or newer:
     Ubuntu 22.04, Debian 12, Fedora 36 or later)
2. Extract it; it contains an `AI-Cover-Maker` folder.
   - **Windows**: right click > "Extract All..." (or "Extract here" with 7-Zip/WinRAR).
   - **Linux**: `tar -xzf AI-Cover-Maker-vX.X.X-linux.tar.gz`
3. Run `run-AI-Cover-Maker.bat` (Windows) or `./run-AI-Cover-Maker.sh` (Linux).

FFmpeg is still needed, and Git is needed to update it.

### Google Colab
Open the [Colab](https://colab.research.google.com/github/Eddycrack864/AI-Cover-Maker/blob/main/AI_Cover_Maker_Colab.ipynb),
run the **Installation** cell and then the **Run UI** cell, and open the public link it prints.
- With *use_drive* checked, your voice models, separation models and covers are kept in `MyDrive/AI-Cover-Maker`,
  so they are still there the next time.
- If the Gradio link doesn't work, choose another **Tunnel** (Ngrok, Cloudflare, LocalTunnel or Horizon).

## Running AI Cover Maker
- **Windows**: double-click `run-AI-Cover-Maker.bat`.
- **Linux**: `./run-AI-Cover-Maker.sh`

The UI opens in your browser at `http://127.0.0.1:7755`. Extra options can be added to the run scripts (or to
`python app.py`):

| Option | Description |
|---|---|
| `--share` | Create a temporary public Gradio link |
| `--listen-port 8080` | Use another port (default: 7755) |
| `--server-name 0.0.0.0` | Make the UI reachable from other devices on your network |
| `--open` | Open the UI in your browser (the run scripts already do this) |

Example: `run-AI-Cover-Maker.bat --share --listen-port 8080`

To check what AI Cover Maker runs on (GPU, CUDA, FFmpeg), run `info/status-checker.bat` (Windows) or
`info/status-checker.sh` (Linux).

## Updating AI Cover Maker
- **Windows**: double-click `AI-Cover-Maker-updater.bat`.
- **Linux**: `./AI-Cover-Maker-updater.sh`

The updater downloads the latest version from GitHub (Git is needed) and updates the Python environment.
Your settings (`user_data/`), voice and separation models (`models/`) and covers (`outputs/`) are kept. Any change you
made to the files of AI Cover Maker itself is discarded (the updater tells you before doing it).

## How a cover is made
Press **Make cover!** and AI Cover Maker runs these steps. The ones in brackets only run if you choose them:

1. **Vocals / instrumental**: separates the vocals from the instrumental.
2. [**Lead / backing vocals**]: splits the vocals into lead vocals and backing vocals.
3. [**De-reverb**], [**De-echo**], [**Denoise**]: clean the lead vocals.
4. **Voice conversion**: converts the lead vocals with your voice model (RVC).
5. [**Backing vocals**]: converts the backing vocals with a second voice model, or keeps the original ones.
6. [**Key change**]: changes the key of the instrumental and the backing vocals.
7. **Vocal effects** on the converted lead vocals, and the **mix**.

The log under the players shows each step and how long it took. Every step is cached: if you make the same song again
with another voice, other effects or another mix, only the steps that changed are run again.

## Separation steps
Open **Separation steps** to choose the model of each step.

### Presets
| Preset | Vocals | Lead / backing | De-reverb | De-echo | Denoise |
|---|---|---|---|---|---|
| **Fast**: quickest, for a first listen or weak GPUs | MelBand Roformer Kim | — | UVR-DeEcho-DeReverb | — | — |
| **Balanced** (default): good for most songs | MelBand Roformer Kim | Mel-Roformer Karaoke Aufr33-Viperx | MelBand Roformer De-Reverb by anvuew | — | — |
| **Best quality**: every cleanup step with the strongest models, slowest | BS Roformer SW | BS Roformer Karaoke by anvuew | MelBand Roformer De-Reverb by anvuew | UVR-De-Echo-Normal | Mel-Roformer-Denoise-Aufr33 |

After choosing a preset you can still change any step.

### The steps
| Step | What it does | When to use it |
|---|---|---|
| **Vocals / instrumental model** | Separates the vocals from the instrumental. Required | Always. Roformer models give the cleanest vocals |
| **Lead / backing vocals model** | Splits the vocals into lead and backing vocals. Only the lead vocals are cleaned and converted with your voice | Songs with choirs or harmonies. Without it, the backing vocals are converted too, which often sounds worse |
| **De-reverb model** | Removes the reverb of the lead vocals | Recommended: RVC sounds much cleaner on dry vocals. The vocal effects add a clean reverb back |
| **De-echo model** | Removes echo and delay from the lead vocals | Songs with obvious echo/delay effects on the voice |
| **Denoise model** | Removes noise from the lead vocals | Live recordings or noisy songs |

- Each list only shows the models made for that step. The recommended model of each step is marked with a ⭐ and
  comes first.
- **Show all models**, next to the preset, lists every model of the catalog in every step. A model that doesn't
  output what the step needs (e.g. an instrumental model in the vocals step) stops the cover with an error.
- A model is downloaded to `models/separation/` the first time you use it.
- The separation parameters are always UVR5 UI's defaults for each model, so there is nothing to tune.

## Voice conversion settings (RVC)
Open **Voice conversion settings**.

| Setting | Default | What it does |
|---|---|---|
| **Pitch** | 0 | Semitones added to the vocals. Use +12 for a male voice with a female model and -12 for the opposite. A wrong pitch is the most common reason for a bad cover |
| **Pitch extraction method** | rmvpe | How the melody of the vocals is detected. **rmvpe** is the best for singing; **fcpe** is fast and good too; **crepe** is slow and smooth; **swift** is the fastest |
| **Search feature ratio** | 0.75 | How much the `.index` file of the voice model is used. Higher sounds more like the model's accent, but adds artifacts. Lower it if the voice sounds metallic. It does nothing if the voice model has no `.index` |
| **Protect voiceless consonants** | 0.33 | Protects breaths and consonants (s, t, k...) from being converted, which avoids artifacts. Lower protects more; 0.5 turns it off |
| **Volume envelope** | 0.25 | 0 makes the converted voice follow the loudness of the original singer, 1 keeps the loudness of the converted voice |

**More conversion settings**:

| Setting | Default | What it does |
|---|---|---|
| **Autotune** / **Autotune strength** | off / 1 | Snaps the pitch to the nearest note. Useful for voice models that sing out of tune |
| **Automatic pitch** / **Voice model range (Hz)** | off / 155 | Moves the singing into the range of the voice model automatically. Set the range to about 155 Hz for male models and 255 Hz for female models |
| **Embedder model** | contentvec | Must be the one the voice model was trained with. Almost all models use **contentvec**; others are spin, spin-v2, and Chinese, Japanese or Korean HuBERT. **custom**: a folder with `pytorch_model.bin` and `config.json` |
| **Formant shifting** / **Quefrency** / **Timbre** | off / 1 / 1 | Changes the timbre of the voice without changing its pitch, e.g. to make a male to female conversion sound more natural |
| **Noise cleanup** / **Cleanup strength** | off / 12 | Reduces noise in the converted voice. **light** is safe for singing; **speech** is stronger and may remove parts of the singing |

The pitch extraction models and embedders are downloaded to `models/rvc_assets/` the first time they are needed.

## Backing vocals
Needs a **Lead / backing vocals model** in the separation steps.
- **Backing vocals voice model**: converts the backing vocals with this voice. Leave it on *Keep the original backing
  vocals* to keep the original singer.
- **Backing vocals pitch**: the pitch of the backing vocals' voice, in semitones (the lead vocals' pitch is not used).
  The other conversion settings are the same as the lead vocals'.

## Vocal effects
Effects applied to the converted lead vocals. Each effect has a checkbox, and its settings only apply while it is
checked. The defaults are the effects of [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen).

| Effect | Settings (default) | What it does |
|---|---|---|
| **High-pass filter** | Cutoff frequency (50 Hz) | Removes rumble below the cutoff |
| **Compressor** | Threshold (-15 dB), Ratio (4) | Evens out the vocal volume: the parts louder than the threshold are lowered by the ratio |
| **Reverb** | Room size (0.15), Damping (0.7), Wet level (0.2), Dry level (0.8), Width (1) | Puts the dry converted vocals back in a room. **Room size**: how big the room is. **Damping**: how much the high frequencies of the reverb are absorbed. **Wet/Dry level**: how much reverb and how much of the original voice. **Width**: stereo width of the reverb |

## Mix
| Setting | Default | What it does |
|---|---|---|
| **Change the key of the song** | 0 | Semitones for the whole song: the instrumental, the backing vocals and the converted vocals, keeping the tempo. Use it when the song is too high or too low for the voice model |
| **Lead vocals volume (dB)** | 0 | Volume of the converted lead vocals |
| **Backing vocals volume (dB)** | 0 | Volume of the backing vocals |
| **Instrumental volume (dB)** | 0 | Volume of the instrumental |

The volumes are added to the base levels of AICoverGen (-4 dB lead vocals, -6 dB backing vocals, -7 dB instrumental),
so 0 dB already gives a balanced mix. A limiter keeps the cover below -1 dBFS, so it never clips.

**Select the output format** (next to the voice model): wav, flac, mp3, ogg or m4a, for the cover and every stem saved
with it.

## Batch covers
Open **Batch covers**, write the path of a folder with songs in **Input path** and, optionally, an **Output path**
(by default, `outputs/`), and press **Make all the covers!**. Every song is made with the current settings, and the
log shows each one as it finishes. **Stop** cancels the batch.

## Output files and the cover cache
Every cover is saved in its own folder, `outputs/<song> (<voice> Cover)/`, with every stem made along the way, all named
`<song> (<stem>).<format>`. For example:

- `My Song (Atri Cover).wav`: the cover.
- `My Song (Instrumental).wav`, `My Song (Vocals).wav`
- `My Song (Lead Vocals).wav`, `My Song (Backing Vocals).wav`
- `My Song (Lead Vocals, No Reverb).wav`, `My Song (Reverb).wav`
- `My Song (Lead Vocals, Atri).wav`: the converted vocals.

Making the same cover again replaces the files of that folder.

Every step is also kept in `workspace/` (the **cover cache**), so trying other voices, effects or mixes on the same song
is fast. It can take a lot of space: clear it from **Settings > Cover cache**.

## Voices tab
- **Download a voice model by link**: links to Hugging Face, Google Drive (public files) or a direct link to a `.zip`,
  `.pth` or `.index` file. **Name** is optional; by default it is the file name.
- **Upload a voice model**: a `.pth` file plus, optionally, its `.index` file, or a `.zip` with both.
- **Delete a voice model**.

Voice models are kept in `models/rvc/<name>/`: a `.pth` file and, optionally, an `.index` file. You can also copy them
there yourself and press **Refresh voice models**. A folder with several `.pth` files lists one voice per file, as
`<folder>/<file>`, sharing the folder's `.index`.

## Settings tab
- **Language selector**: the language of the UI (19 languages), or the system's language. Restart AI Cover Maker to
  apply it.
- **Save settings**: saves every option of the Cover tab (preset, models, conversion settings, effects, mix...) so they
  are loaded every time AI Cover Maker starts, or goes back to the default settings. They are saved in
  `user_data/cover_settings.json`.
- **Cover cache**: shows the size of `workspace/` and clears it.
- **Discord Rich Presence**: shows what you are doing in AI Cover Maker on your Discord profile.

The **Themes** tab changes the look of the UI.

## Folders
| Folder | What it has |
|---|---|
| `models/rvc/` | Your voice models |
| `models/separation/` | Separation models, downloaded when first used |
| `models/rvc_assets/` | Pitch extraction models and embedders, downloaded when first used |
| `outputs/` | Your covers |
| `workspace/` | The cover cache |
| `user_data/` | Your settings (language, theme, saved Cover settings) |
| `env/` | The Python environment made by the installer |

None of these folders are touched by the updater, except `env/`, which it updates.

## Tips for better covers
- **Get the pitch right first**: +12 or -12 when the singer and the voice model are of different genders, or use
  **Automatic pitch**. If the song is simply too high or too low for the voice, use **Change the key of the song**.
- **Use a lead / backing vocals model** for songs with harmonies, and **de-reverb**: RVC converts dry, single voices
  best.
- **Metallic or robotic voice**: lower the **Search feature ratio** (e.g. 0.5) and the **Protect voiceless consonants**
  value (e.g. 0.25).
- **The voice is too loud or too quiet**: change the **Lead vocals volume**, or the **Volume envelope**.
- **Same song, many voices**: the separation is only done once, so the second cover is much faster.

## Contributions
Contributions are welcome! Feel free to:
- Report issues or bugs via the [issue tracker](https://github.com/Eddycrack864/AI-Cover-Maker/issues).
- Submit improvements through [pull requests](https://github.com/Eddycrack864/AI-Cover-Maker/pulls). See the
  [Development section of the README](https://github.com/Eddycrack864/AI-Cover-Maker#development) for the project
  layout and how to run the tests.

Star the repository if you find it useful.

## Credits
See the [Credits section of the README](https://github.com/Eddycrack864/AI-Cover-Maker#credits).

## Feedback and Support
- **Troubleshooting**: check the [troubleshooting guide](troubleshooting.md).
- **Community**: join the [AI HUB Discord](https://discord.gg/aihub) for support and discussions.
