# AI-Cover-Maker — Feasibility & Target Architecture Plan

## Context

`Eddycrack864/RVC-AI-Cover-Maker-UI` (v1.0.5, last push 2025-08) automates AI song covers:
separate → RVC-convert lead (and optionally backing) vocals → effects → remix. It is unmaintained and
fragile: Python 3.9, torch 2.1.1, Gradio 4.36, a 230-line `pip freeze` installed with `--no-deps`,
vendored Applio ~3.2 plus vendored ZFTurbo MSST run as `python` subprocesses, and several silent bugs
(autotune never runs because `== "True"` is compared against a bool; instrumental pitch shift is never
applied and also changes tempo; the "Agggressive" typo; the FoxJoy and denoise branches are broken; the
filename-keyword step cache reuses stale outputs; the mix can clip).

Goal: rewrite from scratch in this repo (`AI-Cover-Maker`):
- **Applio v3.7.x** (MIT) for RVC inference.
- **UVR5-UI v2.0** (MIT, same author) as the app skeleton: layered package, Gradio 6, spec-driven
  tabs, config store, i18n, themes, resumable model downloader, installers/updaters, Colab notebook.
- **python-audio-separator** for every separation step. It natively supports Roformer/MDX23C/VR/MDX/
  Demucs, so the vendored MSST is dropped.

User decisions:
- Vendor an inference-only subset of Applio.
- NVIDIA CUDA 12.8 first (keep GTX support if possible), CPU fallback, MPS experimental.
- Cloud v1 = Google Colab only.
- English code, docs and CLAUDE.md; UI i18n en_US + es_ES (+ pt_BR carried over).

## Feasibility findings

| Area | UVR5-UI (main) | Applio (main, v3.7.1) | Verdict |
|---|---|---|---|
| Python | 3.13.16 (conda-forge) | 3.12 | Target **3.12** (Colab runtime is 3.12; Applio-tested). Spike also on 3.13. |
| torch | 2.7.1 cu128 | 2.11.0 cu128 | Target **2.7.1 cu128**. Hypothesis to verify: torch ≥2.8 cu128 wheels dropped Pascal (GTX 10xx); 2.7.1 cu128 covers Pascal→Blackwell. Applio inference only needs `torch.load(weights_only=True)` + standard ops. Fall back to newer torch only if the spike fails. |
| Gradio | 6.30.0 | 6.20.0 | **6.30.0** (we use UVR5-UI's UI code, not Applio's). |
| numpy | unpinned (2.x) | 2.4.6 | numpy 2.x OK for both. |
| librosa | `<1.0` (audio-separator 0.47 imports audioread) | 0.11.0 | **0.11.x**. |
| onnxruntime | `onnxruntime-gpu<1.27` (CUDA 12) | `onnxruntime` / `-gpu==1.22` (Win) | Install **only `onnxruntime-gpu<1.27`**. Applio's `Swift.py` imports onnxruntime at import time, so one runtime must be present. Never both. |
| transformers | — | 5.13.1 | Check it accepts torch 2.7.1 (spike). Pin a lower 4.x/5.x if not. |
| Other RVC deps | — | faiss-cpu 1.14.3, torchcrepe 0.0.24, torchfcpe 0.0.4, soxr, stftpitchshift, einops | Check wheels exist for py3.12 win/linux (and macOS arm64). |
| Effects | — | removed pedalboard | Add **pedalboard** for reverb/compressor/EQ/limiter/pitch shift. Check wheels. |
| Not needed | | fairseq, noisereduce, tensorboard, matplotlib, edge-tts, sounddevice, webrtcvad, fastapi/uvicorn (realtime), click | Excluded. |

Conclusion: **feasible**. The risks are dependency resolution (torch 2.7.1 × transformers 5 × audio-separator
0.47) and GPU memory when separation and RVC models share one process. Both are handled in Phase 1
before any UI work.

## Target architecture

```
app.py                         # thin launcher -> ai_cover_maker.main.main()
ai_cover_maker/
  main.py cli.py paths.py i18n.py            # from UVR5-UI, renamed; port default 7755 (old app)
  config/ store.py settings.py               # ConfigStore (default+user_data merge, atomic writes), presets
  core/
    errors.py device.py audio.py             # device: cuda > mps (experimental, opt-in) > cpu
    downloads/ model_downloader.py ytdl.py   # resumable + mirror fallback; yt-dlp -> wav
    models/ catalog.py                       # separation catalog (assets/models.json, UVR5-UI format)
    separation/ runtime.py service.py        # Separator cache + lock (UVR5-UI)
    rvc/ engine.py voices.py assets.py       # adapter over vendored Applio; voice model mgmt; predictor/embedder downloads
    effects/ chain.py                        # pedalboard chains (vocal FX, reverb, limiter), key shift
    mix/ mixer.py                            # sample-rate align, pad, dB gains, true-peak limiter, export
    pipeline/ stages.py cache.py runner.py   # Cover pipeline (see below)
  ui/ app.py context.py theme.py adapters.py components.py
      tabs/ cover.py separation.py convert.py mixer.py voices.py settings.py themes.py credits.py
third_party/applio/            # vendored inference subset + LICENSE + UPSTREAM (commit sha) + patches/
assets/ models.json recommended_models.json pipeline_presets.json config.default.json i18n/ themes/
tools/ sync_applio.py sync_models.py i18n_scan.py
tests/
models/{separation,rvc,rvc_assets}/  workspace/  outputs/  user_data/   (gitignored)
AI-Cover-Maker-installer.{bat,sh}  AI-Cover-Maker-updater.{bat,sh}  run-AI-Cover-Maker.{bat,sh}
AI_Cover_Maker_Colab.ipynb
.reference/                    # gitignored shallow clones of the 3 reference repos
.claude/CLAUDE.md  .claude/plans/
```

### Applio integration (vendored subset)
- `tools/sync_applio.py --ref <tag>` copies the inference file list (`rvc/configs`, `rvc/infer`,
  `rvc/lib/{utils,user_config,zluda}.py`, `rvc/lib/algorithm/**` minus `discriminators.py`,
  `rvc/lib/predictors/{f0,RMVPE,Swift,chunking}.py`,
  `rvc/lib/tools/{http_download,device,formant,audio_cleanup,gpu_checker,model_download,gdown,prerequisites_download}.py`,
  `assets/config_template.json`, `LICENSE`) from `.reference/Applio` into `third_party/applio/`.
  It then applies `third_party/applio/patches/*.patch` and writes `UPSTREAM`.
- Patches stay minimal: resolve predictor/embedder paths from an env root instead of `os.getcwd()`,
  and drop the `sys.path.append(os.getcwd())` lines.
- `core/rvc/engine.py`:
  - Before any import, set `APPLIO_ROOT`, `APPLIO_CONFIG_FILE` (into `user_data/`) and
    `APPLIO_LOGS_DIR` (→ `models/rvc`), and add `third_party/applio` to `sys.path`.
  - Expose `convert(input, output, voice, params) -> Path`, wrapping
    `VoiceConverter.convert_audio(...)` with typed `RvcParams`: pitch, f0_method
    (rmvpe/crepe/fcpe/swift), index_rate, volume_envelope, protect, f0_autotune(+strength),
    proposed_pitch, embedder (+custom), formant shift, cleanup_mode, sid.
  - Plus `unload()` for GPU memory.
- Assets: rmvpe.pt, fcpe.pt, contentvec, from `huggingface.co/IAHispano/Applio/resolve/main/Resources/`.
  Download them lazily with our resumable downloader. Do not run Applio's prerequisites pipeline.

### Cover pipeline (`core/pipeline`)
Each stage is a declarative spec: `id`, input stems, model/params, output stems.
1. Input: upload, file path, URL (yt-dlp), or batch folder.
2. Vocal/instrumental separation (Roformer default).
3. Optional lead/backing split (karaoke Roformer).
4. Optional de-reverb / de-echo.
5. Optional denoise.
6. RVC on lead vocals; optional second voice/params for backing vocals.
7. Vocal FX: reverb/compressor/EQ (pedalboard).
8. Key change: one setting drives RVC pitch and a formant-preserving instrumental pitch shift (pedalboard `PitchShift`), keeping tempo.
9. Mix: per-stem dB gains, sample-rate/length alignment, limiter. Export wav/flac/mp3/ogg/m4a. Optionally keep intermediate stems + zip.

Rules for the pipeline:
- **Cache key = hash(input-file content hash, stage id, model id, params, upstream keys).** Stored in
  `workspace/<song-hash>/<stage>-<key>.wav` with a manifest JSON. This fixes the stale-cache bug, and
  re-running with only mix changes is instant.
- Single process, no subprocesses. Use a global GPU lock and concurrency id (UVR5-UI pattern). Unload
  the Separator before loading RVC when VRAM ≤ 8 GB.
- Progress goes through `progress_callback` with stage-weighted fractions. Errors are typed
  (`SeparationError`, `ConversionError`, `MixError`) and shown via the `user_errors` decorator.
- Default stage models come from `assets/pipeline_presets.json` ("Fast", "Balanced", "Best quality").
  Their ids must exist in `assets/models.json` (copied from UVR5-UI).

### UI tabs
- **Cover**: the full pipeline, with presets, accordions per stage, and batch mode.
- **Separation**: a simplified single-model run that reuses UVR5-UI `ArchSpec`s.
- **Voice conversion**: RVC only (Applio params), single/batch.
- **Mixer**: remix from cached stems without recomputing.
- **Voices**: download zip/.pth/.index URLs via Applio `model_download` plus direct-file handling, upload, rename, delete, index auto-match.
- **Settings**, **Themes**, **Credits**.

### Hardware / MPS
- `core/device.py` picks cuda, then mps (only if `config.device.allow_mps` or auto, with a fallback env
  `PYTORCH_ENABLE_MPS_FALLBACK=1`), then cpu. It shows the effective device in Settings.
- Testing MPS without a Mac:
  1. CI on GitHub `macos-14` (arm64): install script, import smoke, and a CPU end-to-end test on a
     5-second clip. GitHub VMs expose no MPS, so this proves packaging only.
  2. Before each release, a short manual MPS session on a rented Apple-silicon Mac (Scaleway Mac mini
     or AWS EC2 mac2, billed hourly) using a scripted `tools/smoke_test.py --device mps`.
  3. Mark MPS "experimental" in the README and ask for community reports.

## Phases

**Phase 0 — Bootstrap (first action after approval)**
- `.gitignore` (`.reference/`, `env/`, `models/`, `workspace/`, `outputs/`, `user_data/`, …).
- `git clone --depth 1` the three repos into `.reference/{RVC-AI-Cover-Maker-UI,Applio,UVR5-UI}`.
- Write `.claude/CLAUDE.md`: purpose, reference-repo locations and roles, the architecture above,
  the vendoring rule ("never edit `third_party/applio` by hand; use patches + sync script"), stack
  pins, and commands (installer, run, pytest, sync tools) marked TBD until they exist.
- Copy this plan to `.claude/plans/00-feasibility-and-architecture.md`. Later phases get their own plan files.

**Phase 1 — Dependency spike (go/no-go)**
- Draft `requirements.txt`: `audio-separator[gpu]==0.47.0`, `onnxruntime-gpu<1.27`, `librosa<1.0`,
  `gradio==6.30.0`, `yt_dlp`, `pedalboard`, `faiss-cpu`, `torchcrepe`, `torchfcpe`, `transformers`,
  `soxr`, `stftpitchshift`, `einops`, `beautifulsoup4`, `pypresence`. Torch is installed separately
  (2.7.1 cu128).
- Resolve with `uv pip compile` for py3.12 and 3.13 on win/linux/macOS-arm64, then install into a
  fresh env.
- Script `tools/smoke_test.py`: separate a 30-second clip (Roformer), RVC-convert with a test voice,
  mix, and print VRAM peaks. Run it on the user's NVIDIA GPU and on CPU.
- Exit criteria: clean resolution, end-to-end smoke passes, documented pins. If torch 2.7.1 fails,
  go to torch 2.11 cu128 and document dropping GTX 10xx.

**Phase 2 — Skeleton**: port UVR5-UI `paths/config/i18n/theme/adapters/cli/main`, the catalog, the
downloader and ytdl. Add pytest copied/adapted from UVR5-UI.

**Phase 3 — Engines**: `separation/` (UVR5-UI), `rvc/` (vendored Applio + adapter + voice manager),
`effects/`, `mix/`. Unit tests use short synthetic audio, and the slow GPU tests are behind a marker.

**Phase 4 — Pipeline**: stages, content-hash cache, runner, presets, batch.

**Phase 5 — UI**: tabs listed above, plus i18n en_US/es_ES/pt_BR.

**Phase 6 — Distribution**: installer/updater/run scripts adapted from UVR5-UI (Miniconda +
conda-forge python 3.12, torch cu128, CPU-only option, macOS `.sh` branch), the Colab notebook
(tunnels: Gradio/Ngrok/Cloudflare/LocalTunnel, optional Drive mount for models/outputs), README,
credits (Applio, audio-separator, UVR, model authors), and CI (lint + pytest on ubuntu/windows/macos-14).

## Verification
- Phase 1: `tools/smoke_test.py` passes on the local CUDA GPU and on CPU. `uv pip compile` succeeds
  for all target platforms.
- Each phase: `pytest` (fast suite) green. `pytest -m gpu` runs the end-to-end cover on a fixture song.
- UI: launch `run-AI-Cover-Maker.bat --open`, make one cover with each preset, then change only the
  mix gains and confirm separation/RVC are served from cache (log shows cache hits). Check that the
  key change shifts the instrumental without changing its duration.
- Colab: run the notebook on a T4 runtime and complete one cover through a Gradio share link.
- Regressions vs the old app: autotune changes output, instrumental key shift is applied, no clipping
  (peak ≤ -0.1 dBFS after the limiter).
