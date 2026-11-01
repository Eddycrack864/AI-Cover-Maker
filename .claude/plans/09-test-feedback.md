# 09 — Feedback after testing covers (done)

1. `rmvpe+`: Applio 3.7.x has no such method. Its pipeline implements `crepe`, `rmvpe`, `fcpe` and
   `swift`, which is what `F0_METHODS` offers. Applio's `core.py` CLI still lists `hybrid[...]` choices,
   but the pipeline has no branch for them, so they are not offered here.
2. New `assets/favicon.ico` from the user.
3. "Search feature ratio" (`RvcParams.index_rate`) defaults to 0.75.
4. The cover's progress bars are on every audio output: Cover, Instrumental and Backing vocals (when
   visible). "Output information" stays a plain streaming log. Re-runs served from the cache (RVC or
   mix only) show the bars too. Two Gradio behaviours had hidden them:
   - Each log yield puts the outputs in "generating" state, which has no bar, until the next progress
     report. RVC reports no progress while it converts, so `run_streaming` repeats the last report
     after every yield.
   - A bar is not drawn over a player that has audio, so a js-only event clears the players before
     the cover event runs (`clear.then(run)`).
   Verified in the browser: full run (2 bars, 3 once backing vocals show) and an RVC-only re-run (3 bars).
5. "Refresh voice models" and "Clear the cache" are primary buttons.
6. Removed the "Performance" settings: precision is always auto (autocast on CUDA, FP32 otherwise).
   audio-separator has no automatic mode of its own (it defaults to FP32), and passing autocast on
   CPU would turn on CPU autocast. `RuntimeOptions.from_config` and the `performance` config keys are gone.
7. Removed the "Alternative model downloader" settings. `ensure_downloaded` tries audio-separator's own
   download, then GitHub, then Hugging Face. audio-separator writes to the final file name, so
   `runtime.atomic_downloads` makes it write to `<file>.download` and rename it. Our downloads resume
   from `.part`; a pending `.part` skips audio-separator. The wget/curl methods were removed.
8. Deleting `<folder>/<checkpoint>` deletes only that `.pth`; the shared index stays until the last
   checkpoint is deleted, which deletes the folder.
