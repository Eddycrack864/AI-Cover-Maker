# Phase 2 — App skeleton (done)

Ported from UVR5-UI at 4f7d149 into `ai_cover_maker/`. The package was renamed (`uvr5_ui` →
`ai_cover_maker`), along with `UVR5_UI_ROOT` → `AI_COVER_MAKER_ROOT` and `UVRError` → `AppError`.

## Kept

- `paths`, `cli` (default port **7755**), `main`, `i18n`, `config/{store,settings}`.
- `core/{audio,device,errors}`, `core/downloads/{model_downloader,ytdl}`, `core/models/catalog`.
- `core/separation/{specs,runtime,service}`.
- `ui/{app,context,adapters,components,theme,a11y,stem_mixer}`.
- Tabs: separation (the 5 architecture tabs are nested under a "Separation" tab), themes, settings,
  credits.
- `assets/`: `models.json`, `recommended_models.json`, `default_settings.json`,
  `config.default.json`, themes list, and i18n for en_US, es_ES, pt_BR. The favicon is the old
  cover maker logo.
- `tools/`: `i18n_scan`, `sync_models`, `check_model_links`, `migrate_models_json`.
- The pytest suite, adapted.

## Dropped

- Ensemble (core, presets, tab, settings section), the Leaderboard, and Discord presence (it would
  need its own Discord app id).
- The `info/docs.md` recommended-models test.

## Changed

- `paths`:
  - Separation weights go in `models/separation/`.
  - New: `VOICE_MODELS_DIR` (`models/rvc`), `RVC_ASSETS_DIR`, `WORKSPACE_DIR`, `APPLIO_DIR`,
    `APPLIO_CONFIG_FILE`.
- `errors`: added `ConversionError` and `MixError`.
- Fixed in `a11y.announce`: the announcement id was `time.monotonic_ns()`. Windows' clock resolution
  is about 15 ms, so two quick announcements could get the same id (a test caught this). It is now a
  counter. **The same bug exists upstream in UVR5-UI.**

## Verification

- `python -m pytest -q`: 99 passed, 3 skipped (network).
- `python app.py --listen-port 7756`: serves HTTP 200.
