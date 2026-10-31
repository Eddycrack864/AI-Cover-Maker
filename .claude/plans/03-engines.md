# Phase 3 — Engines (done)

## Modules

- `core/device.py`: `gpu_lock`, an `RLock`. It is shared by separation (`runtime.separation_lock`
  is an alias) and RVC, so the pipeline can hold it across stages.
- `core/rvc/engine.py`:
  - `_setup_applio()` sets the `APPLIO_*` env vars and adds `third_party/applio` to `sys.path`.
    `import_applio(module)` calls it before importing.
  - `RvcParams` is a frozen dataclass mirroring `convert_audio` kwargs, with `validate()`.
  - `RvcEngine.convert(input, output, voice, params, progress)`:
    - Downloads missing assets, then runs under `gpu_lock`.
    - Writes to a `.part.wav` and renames it.
    - Wraps any failure in `ConversionError` and unloads the model.
  - `unload()` calls `cleanup_model` and empties the CUDA cache. `rvc_engine` is the process-wide
    instance.
- `core/rvc/assets.py`:
  - RMVPE/FCPE/Swift predictors and HF embedders, downloaded with UVR5-UI's resumable
    `_download_python`.
  - `required_files`, `missing_files`, `ensure_assets`.
- `core/rvc/voices.py`:
  - Layout is `models/rvc/<folder>/*.pth|.onnx` plus an index. `G_*`/`D_*` and `trained*` indexes
    are skipped, and `added*` is preferred.
  - A folder with several checkpoints lists each one as `<folder>/<stem>`.
  - `install_voice` copies files or zips; zip extraction rejects path traversal.
  - `download_voice` uses Applio's `download_from_url` only for the fetch (Google Drive, HF), with
    `zips_path` swapped to a temp dir under a lock, because it chdirs. Our code then installs the
    files, so bare `.pth` links work too.
  - `delete_voice` refuses to delete anything outside the voices folder.
- `core/effects/chain.py`:
  - `VocalEffects`: high-pass, compressor and reverb, all off by default.
  - `apply_effects`.
  - `pitch_shift`: pedalboard `PitchShift`, which keeps the duration.
- `core/mix/mixer.py`:
  - `mix_waves` pads and upmixes to stereo, applying dB gains.
  - `limit` runs a pedalboard Limiter and then a hard peak ceiling (-1 dBFS default).
  - `mix_tracks` resamples to the first track's sample rate and exports in any `core/audio`
    format.

## Tests

- `tests/test_rvc.py`: voices, install/delete, assets, and the engine with a fake converter. It also
  checks that autotune reaches Applio as a real bool.
- `tests/test_mix.py`: padding/upmix, the limiter ceiling, resampled mixing, pitch shift (same
  length, octave up = 2x frequency), and effects.
- `tests/test_integration.py` (`--slow`): real Applio conversion with the local Atri voice. It
  passed on CPU in about 38 s.

Totals: fast suite 112 passed. Slow suite 1 passed.
