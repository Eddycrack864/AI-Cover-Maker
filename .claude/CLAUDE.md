# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI-Cover-Maker is a from-scratch rewrite of the unmaintained
[RVC-AI-Cover-Maker-UI](https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI). It is a Gradio app that
makes an AI song cover automatically:
1. Separate the song with python-audio-separator.
2. Optionally split lead and backing vocals and clean the lead.
3. Convert the vocals with RVC (vendored Applio inference).
4. Apply effects and remix.

It installs locally with Windows/Linux scripts and runs on Google Colab.

- Work on the `dev` branch, not `main`.
- MPS (Apple) support is postponed.
- Code, comments, docs and commit messages are in English. The UI is translated through i18n
  (en_US, es_ES, pt_BR).
- History and decisions are in `.claude/plans/`:
  - `00-feasibility-and-architecture.md` is the approved plan.
  - `01`–`06` are per-phase notes, `07`+ later changes.
  - Add a new numbered file for each new piece of work.

## Reference repositories (`.reference/`, gitignored)

These are shallow clones kept locally so GitHub doesn't need to be queried.
- Never import from `.reference/` at runtime and never commit it.
- Refresh with `git -C .reference/<repo> pull --depth 1`.

| Folder | Role |
|---|---|
| `UVR5-UI/` | Source of the app skeleton: config, i18n, themes, a11y, separation tabs, catalog, downloader, scripts, Colab. Same author. Port fixes both ways when relevant. |
| `Applio/` | Source of `third_party/applio/` (inference only). |
| `RVC-AI-Cover-Maker-UI/` | The old app. Use it only for feature scope; its code has known bugs. |

## Architecture

The package is `ai_cover_maker/`; `app.py` is a thin launcher.

`core/` has no Gradio imports, and modules import torch and audio_separator lazily (a test checks this):
- `separation/`:
  - Ported from UVR5-UI: `ArchSpec`s per architecture and `SeparatorCache`, which keeps the last
    model loaded. Its cache key includes `output_dir`.
  - Models come from `assets/models.json` and are downloaded to `models/separation/`.
  - There is no separation UI. Parameters are always UVR5 UI's defaults from
    `assets/default_settings.json` (`config/settings.py`); users can't change them.
- `rvc/engine.py`:
  - `RvcEngine` wraps Applio's `VoiceConverter`. `_setup_applio()` sets the `APPLIO_*` env vars and
    `sys.path` before any `import rvc`; always go through `import_applio()`.
  - `RvcParams` mirrors the `convert_audio` kwargs.
- `rvc/voices.py`: voice models in `models/rvc/<folder>/`. A folder with several `.pth` files lists
  one voice per file as `<folder>/<stem>`.
- `rvc/assets.py`: predictor and embedder weights in `models/rvc_assets/`, downloaded resumably.
- `effects/`, `mix/`: pedalboard vocal chain, tempo-preserving pitch shift, stereo mix with a -1 dBFS
  limiter.
- `pipeline/`:
  - `CoverPipeline.run()` chains vocals → [karaoke] → [dereverb] → [deecho] → [denoise] → rvc_lead →
    [rvc_backing | shift_backing] → [shift_instrumental] → mix.
  - Each stage is cached in `workspace/<song>-<content hash>/<stage>-<key>/`. The key hashes the
    model, its params and the **upstream stage's key**. Keep that chain intact when adding stages;
    never cache by file name.
  - `roles.py` picks the kept stem by normalized label, because separation models name stems
    inconsistently.
  - Every cover goes to `outputs/<song> (<voice> Cover)/`: the cover plus every stem made along the
    way, all named `<song> (<label>).<output format>` (`KEPT_LABELS`/`REST_LABELS` in `runner.py`).
    A re-run replaces that folder's `<song> (*` audio files.
  - `run(on_log=...)` receives the short, translated step log (`STEP_TEXT`) every time it changes.
- `device.gpu_lock` (an `RLock`) wraps every model load and inference. On GPUs with 8 GB or less
  (`low_vram()`), the separator and the RVC voice are never loaded at the same time.

`integrations/discord_presence.py`: optional Rich Presence (UVR5-UI port). Handlers wrap their work in
`presence.track(<activity>)`. `CLIENT_ID` is the Discord app's Application ID; empty disables it. The
app only has the `logo`, `idling` and `cover` image assets: a new activity needs its image uploaded there.

`ui/` follows UVR5-UI conventions:
- Handlers use `inputs_kwargs`. The parameter names become the public API names, so components they
  would shadow are aliased (e.g. `audio_input`).
- Validators run before queueing.
- GPU events share `concurrency_id=GPU_CONCURRENCY_ID`.
- Every tab starts with `a11y.tab_heading`.
- Theme, css and js go to `launch()` (Gradio 6).
- Tabs: Cover, Voices, Themes, Settings, Credits.
- Long handlers that stream text run the core in a thread with `adapters.run_streaming()`; the
  progress bar goes only on the main output (`show_progress_on`).
- Buttons next to a dropdown or textbox go in `gr.Row(equal_height=True)` (a test checks this).

### Vendored Applio (`third_party/applio/`)
- Never edit these files by hand. `tools/sync_applio.py` copies `INCLUDE` from `.reference/Applio`,
  applies the checked text replacements in `PATCHES`, and writes `UPSTREAM`.
- The current patches remove the cwd `sys.path` hack and resolve weights from `APPLIO_MODELS_DIR`.
- Things that differ from old RVC forks:
  - There is no `split_audio`, `f0_file` or post-processing.
  - `hop_length` is ignored.
  - Inference is fp32.
- `Swift.py` imports onnxruntime at import time. Exactly one onnxruntime package must be installed:
  `onnxruntime-gpu`.

## Stack (verified in Phase 1)

- Python 3.12.
- torch/torchaudio 2.7.1 cu128, chosen to keep GTX 10xx support.
- audio-separator 0.47.0, onnxruntime-gpu <1.27, librosa <1.0, gradio 6.30.0.
- transformers 5.x, faiss-cpu, pedalboard.
- `constraints.txt` pins torch so `pip install --upgrade -r requirements.txt` in the installers can't
  replace the CUDA build. Colab doesn't use it and keeps its own torch.
- FFmpeg must be on PATH.

## Development environment (this machine)

- Never install into the user's global Python.
- Use the standalone uv at `.tools/uv.exe` and the venv `.venv` (Python 3.12 under `.tools/python`).
  Recreate it with:
  ```
  export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
  .tools/uv.exe venv .venv --python 3.12
  .tools/uv.exe pip install --python .venv/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 -r requirements-dev.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  ```
- **No NVIDIA GPU here** (Intel UHD), so everything runs on CPU. CUDA must be verified by the user or
  on Colab.
- Local test assets (gitignored):
  - `.local/samples/test_edited.wav`: 12 s mono clip.
  - `models/rvc/Atri/`: test voice.

## Commands

Use `.venv/Scripts/python.exe` as `python`.
- `python app.py [--open] [--share] [--listen-port 7755] [--server-name 0.0.0.0]`
- Tests:
  - `python -m pytest -q`: fast suite with fakes.
  - `python -m pytest tests/test_pipeline.py -k cache`: a single test.
  - `--slow`: real separation and RVC with the local assets, a few minutes on CPU.
  - `--network`: online catalog checks.
- `python tools/i18n_scan.py --check --langs`
  - `--write` regenerates `en_US.json`.
  - The English string is the key; every language must have exactly the en_US keys.
- `python tools/sync_applio.py`: re-vendor Applio.
- `python tools/smoke_test.py --input <wav> --voice <pth> [--index <index>] [--device cpu|cuda]`:
  dependency-level end-to-end check that does not use the app package.

## Distribution

- `AI-Cover-Maker-installer.{bat,sh}`, `AI-Cover-Maker-updater.{bat,sh}` and `run-AI-Cover-Maker.{bat,sh}`
  are adapted from UVR5-UI.
  - Version variables must match between `.bat` and `.sh`, and must match `constraints.txt`
    (`tests/test_scripts.py`).
  - **Never change the updater `.bat` above its "Do not edit" marker.** Its SHA-256 is pinned in the
    tests.
  - `.gitattributes` keeps `.bat` CRLF and `.sh` LF.
- `AI_Cover_Maker_Colab.ipynb` is adapted from UVR5-UI's notebook.
  - The repo URL and paths are rot13-encoded, as upstream.
  - A `branch` param selects `main` or `dev`.
  - Google Drive is symlinked into `models/` and `outputs/`.
- CI is in `.github/workflows/tests.yml`: pytest and i18n on Ubuntu and Windows with CPU torch.
