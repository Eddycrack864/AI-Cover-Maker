# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI-Cover-Maker is a from-scratch rewrite of the unmaintained
[RVC-AI-Cover-Maker-UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI). It is a Gradio app that
makes an AI song cover automatically: separate the song with python-audio-separator, convert the lead
(and optionally backing) vocals with RVC, apply effects, and remix. It must be installable locally
(Windows/Linux scripts) and runnable on Google Colab.

Status: Phase 1 (dependency spike) is done; see `.claude/plans/01-dependency-spike.md`. The approved
overall plan is `.claude/plans/00-feasibility-and-architecture.md`. Read it before starting a phase, and
add a new plan file in `.claude/plans/` for each later phase.

Work on the `dev` branch, not `main`. MPS support is postponed.

Code, comments, docs and commit messages are in English. The UI is translated through i18n
(en_US, es_ES, pt_BR).

## Reference repositories (`.reference/`, gitignored)

Shallow clones kept locally so GitHub doesn't need to be queried. Never import from `.reference/` at
runtime and never commit it. Refresh with `git -C .reference/<repo> pull --depth 1`. If the folder is
missing, re-clone with `git clone --depth 1 https://github.com/<owner>/<repo>.git .reference/<repo>`.

| Folder | Upstream | Role |
|---|---|---|
| `UVR5-UI/` | Eddycrack864/UVR5-UI | **App skeleton to copy from.** Layered `uvr5_ui/` package, Gradio 6, config store, i18n, themes, model catalog + resumable downloader, yt-dlp, installers/updaters, Colab notebook. Same author, MIT. |
| `Applio/` | IAHispano/Applio | **RVC inference source.** Only the inference subset is vendored (see below). MIT, credit IAHispano. |
| `RVC-AI-Cover-Maker-UI/` | Eddycrack864/RVC-AI-Cover-Maker-UI | Old app. Use it only to recover feature scope and UI options (`core.py`, `tabs/full_inference.py`). Don't copy its code: it has known bugs (string-compared autotune, ignored instrumental pitch, filename-keyword cache). |

## Target architecture

The package is `ai_cover_maker/`, mirroring `uvr5_ui/`. `app.py` is a thin launcher.

- `paths.py`, `config/`, `i18n.py`, `cli.py`, `main.py`: ported from UVR5-UI.
  - All paths resolve from the package location, never the current working directory.
  - User preferences live in gitignored `user_data/` and are merged over `assets/config.default.json`.
- `core/separation/`: UVR5-UI's cached `Separator` plus the global GPU lock. Every separation step
  goes through audio-separator; there is no vendored MSST.
- `core/rvc/`: adapter over the vendored Applio `VoiceConverter`, plus voice-model management and
  predictor/embedder downloads.
- `core/effects/`, `core/mix/`: pedalboard effects, key shift (preserves tempo), and gain/align/limit/export.
- `core/pipeline/`: the Cover pipeline as declarative stages.
  - Each stage output is cached under `workspace/<song-hash>/` with a key hashed from the input content,
    model and params.
  - Never cache by filename keywords.
- `ui/tabs/`: Cover, Separation, Voice conversion, Mixer, Voices, Settings, Themes, Credits.
  - Theme, css and js go to `launch()` (Gradio 6), not `gr.Blocks()`.
  - GPU jobs share one concurrency id.

### Vendored Applio (`third_party/applio/`)
- Never edit these files by hand. Copy them from `.reference/Applio` with
  `python tools/sync_applio.py`. The script holds the file list (`INCLUDE`) and checked text
  replacements (`PATCHES`), and records the upstream commit in `UPSTREAM`. To change vendored
  behaviour, add a patch entry there and re-run the script.
- Keep `LICENSE` alongside the code.
- Applio uses absolute imports (`from rvc.infer.infer import VoiceConverter`). The adapter
  `core/rvc/engine.py` (see `setup_applio_env` in `tools/smoke_test.py`) must do these things before
  importing `rvc`:
  1. Set `APPLIO_ROOT`.
  2. Set `APPLIO_MODELS_DIR` (`models/rvc_assets`), which holds the predictors and embedders and is
     added by our patches.
  3. Set `APPLIO_CONFIG_FILE` (`user_data/applio_config.json`) and `APPLIO_LOGS_DIR` (`models/rvc`).
  4. Put `third_party/applio` on `sys.path`.
- Current Applio API differs from older forks:
  - No `split_audio`, `clean_audio`, `f0_file` or post-process effects.
  - `hop_length` is ignored; `cleanup_mode` replaces noisereduce.
  - f0 methods are `rmvpe`, `crepe`, `fcpe` and `swift`.
  - Inference runs in fp32.
- `Swift.py` imports `onnxruntime` at import time, so the env must contain exactly one onnxruntime
  package: `onnxruntime-gpu`.
- Required assets: `rmvpe.pt`, `fcpe.pt` and contentvec, from
  `https://huggingface.co/IAHispano/Applio/resolve/main/Resources/`.

## Stack targets (to be confirmed by the Phase 1 dependency spike)

- Python 3.12 (Colab's runtime); also test 3.13.
- torch 2.7.1 + CUDA 12.8 (`https://download.pytorch.org/whl/cu128`), chosen to keep GTX 10xx support.
  CPU fallback. MPS is experimental and opt-in.
- `audio-separator[gpu]==0.47.0`, `onnxruntime-gpu<1.27`, `librosa<1.0`, `gradio==6.30.0`.
- Plus Applio's inference deps: faiss-cpu, torchcrepe, torchfcpe, transformers, soxr, stftpitchshift,
  einops. Plus pedalboard and yt-dlp.
- FFmpeg must be on PATH.

## Development environment

- Never install into the user's global Python (3.10).
- Use the standalone uv in `.tools/uv.exe` (gitignored) and the project venv `.venv` (Python 3.12,
  managed by uv under `.tools/python`).
- Recreate the env:
  ```
  export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
  .tools/uv.exe venv .venv --python 3.12
  .tools/uv.exe pip install --python .venv/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  ```
- The dev machine has **no NVIDIA GPU** (Intel UHD only), so everything runs on CPU locally. CUDA has
  to be verified by the user or on Colab.
- Local test assets (gitignored):
  - `.local/samples/test_edited.wav`: 12 s mono 40 kHz clip.
  - `models/rvc/Atri/`: test voice, 3 `.pth` epochs + `added_IVF2604_Flat_nprobe_1_atri_v2.index`.

## Commands

- `python tools/sync_applio.py [--source .reference/Applio]`: re-vendor Applio.
- End-to-end stack check (writes to `.local/smoke/`):
  ```
  python tools/smoke_test.py --input .local/samples/test_edited.wav \
      --voice models/rvc/Atri/atri_e375_s6000.pth \
      --index models/rvc/Atri/added_IVF2604_Flat_nprobe_1_atri_v2.index [--device cpu|cuda]
  ```

Planned, not built yet:
- `run-AI-Cover-Maker.{bat,sh} [--open] [--share] [--listen-port 7755]`
- `AI-Cover-Maker-installer.{bat,sh}`
- `pytest`, with `pytest -m gpu` for slow GPU tests.
