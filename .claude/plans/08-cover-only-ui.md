# 08 — Cover-only UI, stem folders, readable log (done)

Feedback from the user's first test run:

1. The Voice conversion and Separation tabs were removed (`ui/tabs/convert.py`, `ui/tabs/separation.py`,
   `ui/stem_mixer.*`, and their widgets in `components.py`/`a11y.py`). Their API endpoints went with them.
   Tabs left: Cover, Voices, Themes, Settings, Credits.
2. The Cover tab's top row is Voice model | Refresh voice models | Output format.
3. Buttons next to a dropdown or textbox use `gr.Row(equal_height=True)`: Refresh, Download (song link),
   Delete voice model. "Clear the cache" is unchanged, at the user's request.
4. Discord: Application ID `1566198256130719857`. Only the `cover` activity exists, because the app has
   only the `logo`, `idling` and `cover` images.
5. Separation parameters are always `assets/default_settings.json` (UVR5 UI defaults):
   - `SeparationSettings` only has `defaults()`.
   - Removed the save/reset settings accordion, the output name template, and the
     `load_custom_settings`/`output` config keys.
   - `RuntimeOptions.from_config` still reads `output.name_template`, but the pipeline overrides it.
6. "Output information" streams a short translated log ("Removing the reverb... done in 12 s"):
   - `CoverPipeline.run(on_log=...)` and `ui.adapters.run_streaming()`.
   - Only the Cover player shows the progress bar (`show_progress_on`).
7. Every cover goes to `outputs/<song> (<voice> Cover)/`:
   - The folder holds the cover and every stem, named `<song> (<label>)` in the output format.
   - The "Save the stems" checkbox and `CoverOptions.keep_stems` were removed.
   - The Instrumental/Backing players show those named files.
8. The faiss "Could not load library with AVX2 support" lines are harmless. The pip `faiss-cpu`
   wheel for Windows has no AVX2 build, so it loads the generic one. The `faiss.loader` logger is
   now set to WARNING. "Using HiFi-GAN vocoder" is a print inside Applio and is harmless too.

Verified:
- The fast suite passes, and so does `--slow`.
- A real cover through the browser on CPU showed one progress bar and the log growing step by step.
