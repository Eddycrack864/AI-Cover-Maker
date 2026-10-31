# Phase 6 — Distribution (done, partly unverified)

## Files

- `AI-Cover-Maker-installer.{bat,sh}`, `-updater.{bat,sh}`, `run-AI-Cover-Maker.{bat,sh}` and
  `tools/ffmpeg-installer.bat`:
  - Generated from UVR5-UI's scripts with checked replacements.
  - Python 3.12, torch + torchaudio 2.7.1 cu128 (no torchvision).
  - The env check imports `audio_separator.separator, gradio, torch, faiss, pedalboard`.
  - The installers pass `-c constraints.txt`.
- `constraints.txt`: `torch==2.7.1`, `torchaudio==2.7.1`. `uv pip install --dry-run --upgrade -r
  requirements.txt -c constraints.txt` reports "Would make no changes" on the local env.
- `AI_Cover_Maker_Colab.ipynb`:
  - Install cell: Drive option that symlinks `models/` and `outputs/` to
    `MyDrive/AI-Cover-Maker`, plus a `branch` param.
  - Run cell: same tunnels as UVR5-UI (Gradio, Ngrok, Cloudflare, LocalTunnel, Horizon) on port
    7755.
- `README.md`, `LICENSE` (MIT, same as the old app), and `.github/workflows/tests.yml`.
- `tests/test_scripts.py`:
  - pins the updater prefix hash;
  - checks the `.bat`/`.sh` versions match;
  - checks `constraints.txt` matches the installers.

## Not verified on this machine

- The installers were not run: they install Miniconda into the user profile.
- The Colab notebook has not been run, so these are unknown:
  - whether Colab's preinstalled torch works with transformers 5 and audio-separator 0.47;
  - whether `uv pip install` without `--system` works there (it does for UVR5-UI);
  - how long the install takes.
- CUDA in general; the dev machine has no NVIDIA GPU.
- The CI workflow has not run yet; it runs on the first push.
