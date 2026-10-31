# Phase 4 — Cover pipeline (done)

`ai_cover_maker/core/pipeline/`:

- `options.py`: `CoverOptions`.
  - One catalog model id per separation stage; `None` skips that stage.
  - `rvc` / `backing_rvc` (`RvcParams`), `effects`, `key_shift`, per-stem gains, `output_format`,
    `keep_stems`.
  - Presets live in `assets/pipeline_presets.json` (Balanced / Fast / Best quality). A test checks
    that every preset model id exists in `assets/models.json`.
- `roles.py`:
  - Picks the stem to keep from the normalized output labels, best match first, per stage. For
    example "No Reverb"/"noreverb"/"dry", and "no noise" vs "noise".
  - Unknown labels raise a `SeparationError` telling the user to pick another model.
  - Several "rest" stems are summed. For example, a 6-stem model's non-vocal stems become the
    instrumental.
- `cache.py`: `SongWorkspace`.
  - Layout: `workspace/<song>-<sha256[:16]>/<stage>-<key>/` with role-named files.
  - `manifest.json` is written last, so incomplete stages are never reused.
  - The stage key hashes the model, the separation values or RVC params, the voice file
    fingerprint and the upstream stage key.
- `runner.py`: `CoverPipeline.run(song, options, progress)` and `run_batch(dir, options)`.
  - Chain: vocals → [karaoke] → [dereverb] → [deecho] → [denoise] → rvc_lead → [rvc_backing or
    shift_backing] → [shift_instrumental] → effects + mix (the mix is never cached).
  - `key_shift` is added to the RVC pitch and applied to the instrumental and to unconverted
    backing vocals with a tempo-preserving pitch shift. RVC's own pitch only moves the voice.
  - The whole run holds `gpu_lock`.
  - On GPUs with 8 GB or less (`device.low_vram()`), the separator is unloaded before RVC and
    vice versa.
  - The Separator writes to a fixed temp folder, so the loaded model is reused across a batch.
  - Output: `outputs/<song> (<voice> Cover).<fmt>`, plus `(stems)` when `keep_stems` is set.

## Verification

- `tests/test_pipeline.py`, with a fake separator writing real WAVs and a fake RVC engine:
  - full run and a cached re-mix;
  - selective invalidation when the dereverb model changes or RVC params change;
  - key shift (pitch sum, duration kept);
  - backing conversion;
  - 6-stem sum, errors, batch, `split_stems`, workspace behaviour, presets.
- `pytest --slow tests/test_integration.py`: a real Kim Mel-Roformer + Atri RVC cover of the
  12 s sample on CPU. The re-mix reused the `vocals` and `rvc_lead` cache and exported mp3. Both
  runs together took about 4 minutes on CPU.
- Fast suite: 122 passed.
