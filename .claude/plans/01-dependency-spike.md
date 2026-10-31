# Phase 1 — Dependency spike (results)

Date: 2026-10-31. Branch: `dev`.

## Resolution (`uv pip compile`, torch pinned to 2.7.1 via constraint, cu128 extra index)

Resolved cleanly for py3.12 and py3.13 on `x86_64-pc-windows-msvc` and `x86_64-unknown-linux-gnu`.
Resolved versions were identical across all four combinations:

| Package | Version |
|---|---|
| torch / torchaudio | 2.7.1+cu128 |
| audio-separator | 0.47.0 |
| onnxruntime-gpu | 1.26.0 (only onnxruntime in the env) |
| transformers | 5.22.1 (imports and loads contentvec with torch 2.7.1) |
| numpy / scipy / librosa | 2.5.4 / 1.18.1 / 0.11.0 |
| faiss-cpu | 1.15.1 |
| torchcrepe / torchfcpe | 0.0.24 / 0.0.4 |
| pedalboard | 0.9.26 |
| gradio | 6.30.0 |

macOS arm64 was not checked; MPS is postponed by user decision.

## Vendored Applio

- `tools/sync_applio.py` copies 37 files from `.reference/Applio` at 59c1131 (2026-10-28).
- It uses checked text replacements instead of `.patch` files, so no `patch` binary is needed on
  Windows and upstream drift fails loudly.
- Patches:
  - Remove `sys.path.append(os.getcwd())`.
  - Predictors and embedders resolve from `APPLIO_MODELS_DIR` (set to `models/rvc_assets`).
- Verified: there are no writes to `%APPDATA%/Applio` or the cwd when these are set:
  `APPLIO_ROOT`, `APPLIO_MODELS_DIR`, `APPLIO_CONFIG_FILE=user_data/applio_config.json`,
  `APPLIO_LOGS_DIR`.

## Smoke test (`tools/smoke_test.py`)

Setup: CPU (Intel laptop, no NVIDIA GPU on the dev machine), 12.2 s mono 40 kHz clip.
- `vocals_mel_band_roformer.ckpt`: about 70 s.
- RVC (Atri v2 + index, rmvpe, contentvec): about 34 s.
- Mix with pedalboard reverb + limiter: 0.1 s. Output peak -2.57 dBFS.

Notes:
- audio-separator stem labels in filenames are lowercase (`(vocals)`, `(other)`).
- RMVPE and FCPE weights are not auto-downloaded by Applio; the app must fetch them. contentvec is
  auto-downloaded.

## Pending

- CUDA run on an NVIDIA GPU (the user's machine or Colab T4):
  `python tools/smoke_test.py --device cuda ...`. It reports peak VRAM per stage.
- Confirm GTX 10xx (Pascal) works with torch 2.7.1+cu128 if hardware is available.

Verdict: **GO** on the planned stack. The CUDA confirmation is still open.
