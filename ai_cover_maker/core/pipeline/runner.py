"""The cover pipeline: separate -> clean the lead vocals -> RVC -> effects -> remix.

Each stage is cached in the song's workspace (see ``cache``), so re-running with new
mix settings, effects or another voice only recomputes what changed. Every cover is
saved in its own folder together with every stem made along the way.
"""

import logging
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ai_cover_maker import paths
from ai_cover_maker.config.settings import SeparationSettings
from ai_cover_maker.core.audio import AUDIO_EXTENSIONS, DEFAULT_NAME_TEMPLATE, list_audio_files, load_audio, write_audio
from ai_cover_maker.core.device import gpu_lock, low_vram
from ai_cover_maker.core.effects.chain import apply_effects, pitch_shift
from ai_cover_maker.core.errors import AppError, MixError, SeparationError
from ai_cover_maker.core.mix.mixer import BASE_LEVELS_DB, Track, mix_tracks
from ai_cover_maker.core.models.catalog import ModelCatalog, get_catalog
from ai_cover_maker.core.pipeline.cache import SongWorkspace, file_fingerprint, stage_key
from ai_cover_maker.core.pipeline.options import OUTPUT_FORMATS, SEPARATION_STAGES, CoverOptions
from ai_cover_maker.core.pipeline.roles import split_stems
from ai_cover_maker.core.rvc.engine import RvcEngine, rvc_engine
from ai_cover_maker.core.rvc.voices import Voice, get_voice
from ai_cover_maker.core.separation.runtime import RuntimeOptions, SeparatorCache, separator_cache
from ai_cover_maker.core.separation.service import separate_file
from ai_cover_maker.core.separation.specs import ARCHITECTURES_BY_ID
from ai_cover_maker.i18n import N_, i18n

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
LogFn = Callable[[list[str]], None]

# Relative duration of each step, to turn per-step progress into overall progress
STEP_WEIGHTS = {"vocals": 4, "karaoke": 3, "dereverb": 2, "deecho": 2, "denoise": 2,
                "rvc_lead": 3, "rvc_backing": 3, "shift": 1, "mix": 1}
SEPARATION_FORMAT = "wav"  # intermediate stems stay lossless

# What the user reads in the log for each step
STEP_TEXT = {
    "vocals": N_("Separating the vocals from the instrumental"),
    "karaoke": N_("Splitting the lead and backing vocals"),
    "dereverb": N_("Removing the reverb"),
    "deecho": N_("Removing the echo"),
    "denoise": N_("Removing the noise"),
    "rvc_lead": N_("Converting the lead vocals with {voice}"),
    "rvc_backing": N_("Converting the backing vocals with {voice}"),
    "shift_backing": N_("Changing the key of the backing vocals"),
    "shift_instrumental": N_("Changing the key of the instrumental"),
    "mix": N_("Mixing and saving the cover"),
}
# Stem names in the output folder: "<song> (<label>).<format>". "{vocals}" is "Vocals", or
# "Lead Vocals" once the backing vocals are split off.
KEPT_LABELS = {"vocals": "Vocals", "karaoke": "Lead Vocals", "dereverb": "{vocals}, No Reverb",
               "deecho": "{vocals}, No Echo", "denoise": "{vocals}, No Noise"}
REST_LABELS = {"vocals": "Instrumental", "karaoke": "Backing Vocals", "dereverb": "Reverb",
               "deecho": "Echo", "denoise": "Noise"}


def _noop(*args) -> None:
    pass


@dataclass
class CoverResult:
    cover: Path
    folder: Path = None  # the cover and every stem
    stems: dict[str, Path] = field(default_factory=dict)  # final mix tracks by role, inside ``folder``
    log: list[str] = field(default_factory=list)
    cached_steps: list[str] = field(default_factory=list)


class _Progress:
    def __init__(self, steps: list[str], report: ProgressFn):
        total = sum(STEP_WEIGHTS[s] for s in steps) or 1
        self._start, done = {}, 0
        for step in steps:
            self._start[step] = done / total
            done += STEP_WEIGHTS[step]
        self._weight = {s: STEP_WEIGHTS[s] / total for s in steps}
        self._report = report

    def step(self, step: str) -> ProgressFn:
        start, weight = self._start.get(step, 0.0), self._weight.get(step, 0.0)
        return lambda fraction, desc: self._report(min(start + weight * max(fraction, 0.0), 1.0), desc)


class _StepLog:
    """Short, readable log lines ("Removing the reverb... done in 12 s"), sent to ``listener`` as they change."""

    def __init__(self, result: CoverResult, listener: LogFn):
        self.lines = result.log
        self.listener = listener
        self._started = time.monotonic()

    def start(self, step: str, **values) -> None:
        self.lines.append(i18n(STEP_TEXT[step]).format(**values) + "...")
        self._started = time.monotonic()
        self.listener(list(self.lines))

    def done(self) -> None:
        seconds = round(time.monotonic() - self._started)
        self.lines[-1] += " " + i18n("done in {seconds} s").format(seconds=seconds)
        self.listener(list(self.lines))

    def reused(self) -> None:
        self.lines[-1] += " " + i18n("reused from a previous run")
        self.listener(list(self.lines))

    def add(self, line: str) -> None:
        self.lines.append(line)
        self.listener(list(self.lines))


class CoverPipeline:
    def __init__(self, catalog: ModelCatalog = None, separation_settings: dict = None,
                 runtime_options: RuntimeOptions = None, separator: SeparatorCache = None,
                 engine: RvcEngine = None, workspace_root: Path = None, outputs_dir: Path = None):
        self.catalog = catalog or get_catalog()
        # Separation parameters are always the bundled defaults (assets/default_settings.json)
        self.separation_settings = separation_settings if separation_settings is not None else SeparationSettings().defaults()
        # Stems are found by their audio-separator name: a custom name template must not apply here
        self.runtime_options = replace(runtime_options or RuntimeOptions(), name_template=DEFAULT_NAME_TEMPLATE)
        self.separator = separator or separator_cache
        self.engine = engine or rvc_engine
        self.workspace_root = workspace_root
        self.outputs_dir = Path(outputs_dir or paths.OUTPUTS_DIR)

    # -- steps -------------------------------------------------------------------------

    def _separation_values(self, arch: str) -> dict:
        spec = ARCHITECTURES_BY_ID[arch]
        return dict(self.separation_settings.get(spec.settings_section, {}))

    def _separate(self, ws: SongWorkspace, step: str, model_id: str, source: Path, source_key: str,
                  progress: ProgressFn, result: CoverResult, log: _StepLog) -> tuple[Path, Path | None, str]:
        """Run one separation step. Returns ``(kept, rest, key)``; several "rest" stems are summed."""
        model = self.catalog.find(model_id)
        if model is None:
            raise SeparationError(f"Unknown separation model for the {step} step: {model_id}")
        spec = ARCHITECTURES_BY_ID[model.arch]
        values = self._separation_values(model.arch)
        key = stage_key(step, {"model": model.filename, "values": values, "input": source_key})
        log.start(step)
        cached = ws.load(step, key)
        if cached:
            result.cached_steps.append(step)
            log.reused()
            return cached["kept"], cached.get("rest"), key

        progress(0.0, f"{i18n(STEP_TEXT[step])}: {model.id}")
        if low_vram():
            self.engine.unload()
        # A fixed output folder: the loaded Separator is reused only while its settings (output
        # folder included) stay the same, e.g. the vocals model across the songs of a batch
        out_dir = self._tmp_root() / "separator"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        try:
            stems = separate_file(spec, model, str(source), SEPARATION_FORMAT, values, output_dir=out_dir,
                                  progress=progress, cache=self.separator, options=self.runtime_options)
            labelled = {s.label or Path(s.path).stem: s.path for s in stems}
            kept, others = split_stems(step, labelled)
            files = {"kept": Path(kept)}
            if others:
                files["rest"] = _sum_files([Path(p) for p in others], out_dir / "rest_sum.wav")
            stored = ws.store(step, key, files)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        log.done()
        return stored["kept"], stored.get("rest"), key

    def _convert(self, ws: SongWorkspace, step: str, voice: Voice, params, source: Path, source_key: str,
                 progress: ProgressFn, result: CoverResult, log: _StepLog) -> tuple[Path, str]:
        payload = {"model": file_fingerprint(voice.model_path), "index": file_fingerprint(voice.index_path),
                   "params": asdict(params), "input": source_key}
        key = stage_key(step, payload)
        log.start(step, voice=voice.name)
        cached = ws.load(step, key)
        if cached:
            result.cached_steps.append(step)
            log.reused()
            return cached["converted"], key
        if low_vram():
            self.separator.clear()  # both models don't fit in VRAM together
        with tempfile.TemporaryDirectory(prefix=f"{step}-", dir=self._tmp_root()) as tmp:
            out = self.engine.convert(source, Path(tmp) / "converted.wav", voice, params, progress=progress)
            stored = ws.store(step, key, {"converted": out})
        log.done()
        return stored["converted"], key

    def _shift(self, ws: SongWorkspace, step: str, source: Path, source_key: str, semitones: int,
               result: CoverResult, log: _StepLog) -> Path:
        key = stage_key(step, {"semitones": semitones, "input": source_key})
        log.start(step)
        cached = ws.load(step, key)
        if cached:
            result.cached_steps.append(step)
            log.reused()
            return cached["shifted"]
        import soundfile as sf

        sample_rate = sf.info(str(source)).samplerate
        with tempfile.TemporaryDirectory(prefix=f"{step}-", dir=self._tmp_root()) as tmp:
            out = write_audio(Path(tmp) / "shifted.wav", pitch_shift(load_audio(source, sample_rate), sample_rate, semitones),
                              sample_rate)
            stored = ws.store(step, key, {"shifted": out})
        log.done()
        return stored["shifted"]

    # -- run ---------------------------------------------------------------------------

    def run(self, song: str | Path, options: CoverOptions, progress: ProgressFn = _noop,
            on_log: LogFn = _noop) -> CoverResult:
        """Make the cover of ``song``. ``on_log`` receives the readable log every time it changes."""
        song = Path(song)
        if not song.is_file():
            raise AppError(f"Input audio not found: {song}")
        if not options.vocals_model:
            raise SeparationError("Select a model to separate the vocals")
        if options.backing_voice and not options.karaoke_model:
            raise AppError("Converting the backing vocals needs a lead/backing separation model")
        if options.output_format not in OUTPUT_FORMATS:
            raise MixError(f"Unsupported output format: {options.output_format}")
        voice = _resolve_voice(options.voice)
        backing_voice = _resolve_voice(options.backing_voice) if options.backing_voice else None

        steps = [s for s in SEPARATION_STAGES if options.stage_model(s)]
        steps += ["rvc_lead"] + (["rvc_backing"] if backing_voice else []) + (["shift"] if options.key_shift else []) + ["mix"]
        prog = _Progress(steps, progress)
        result = CoverResult(cover=Path())
        log = _StepLog(result, on_log)
        started = time.monotonic()
        # Every file made along the way, by its label in the output folder
        exports: dict[str, Path] = {}

        with gpu_lock:
            ws = SongWorkspace(song, self.workspace_root)
            # Each stem carries the key of the step that produced it, which its consumers hash
            lead, instrumental, vocals_key = self._separate(ws, "vocals", options.vocals_model, song, ws.song_key,
                                                            prog.step("vocals"), result, log)
            if instrumental is None:
                raise SeparationError("The vocals model produced no instrumental stem")
            exports[KEPT_LABELS["vocals"]], exports[REST_LABELS["vocals"]] = lead, instrumental
            vocals_label = KEPT_LABELS["vocals"]
            lead_key, backing, backing_key = vocals_key, None, None
            if options.karaoke_model:
                lead, backing, lead_key = self._separate(ws, "karaoke", options.karaoke_model, lead, vocals_key,
                                                         prog.step("karaoke"), result, log)
                backing_key = lead_key
                vocals_label = KEPT_LABELS["karaoke"]
                exports[vocals_label] = lead
                if backing is not None:
                    exports[REST_LABELS["karaoke"]] = backing
            for step in ("dereverb", "deecho", "denoise"):
                if options.stage_model(step):
                    lead, rest, lead_key = self._separate(ws, step, options.stage_model(step), lead, lead_key,
                                                          prog.step(step), result, log)
                    # Cumulative: "Lead Vocals, No Reverb, No Echo"
                    vocals_label = KEPT_LABELS[step].format(vocals=vocals_label)
                    exports[vocals_label] = lead
                    if rest is not None:
                        exports[REST_LABELS[step]] = rest

            lead_params = replace(options.rvc, pitch=options.rvc.pitch + options.key_shift)
            converted_lead, _ = self._convert(ws, "rvc_lead", voice, lead_params, lead, lead_key,
                                              prog.step("rvc_lead"), result, log)
            lead_label = f"{'Lead Vocals' if options.karaoke_model else 'Vocals'}, {_display_name(voice)}"
            exports[lead_label] = converted_lead

            if backing is not None and backing_voice is not None:
                backing_params = replace(options.backing_rvc, pitch=options.backing_rvc.pitch + options.key_shift)
                backing, _ = self._convert(ws, "rvc_backing", backing_voice, backing_params, backing, backing_key,
                                           prog.step("rvc_backing"), result, log)
                exports[f"Backing Vocals, {_display_name(backing_voice)}"] = backing
            elif backing is not None and options.key_shift:
                backing = self._shift(ws, "shift_backing", backing, backing_key, options.key_shift, result, log)
                exports[f"Backing Vocals, Key {options.key_shift:+d}"] = backing
            if options.key_shift:
                prog.step("shift")(0.0, i18n(STEP_TEXT["shift_instrumental"]))
                instrumental = self._shift(ws, "shift_instrumental", instrumental, vocals_key, options.key_shift,
                                           result, log)
                exports[f"Instrumental, Key {options.key_shift:+d}"] = instrumental

        log.start("mix")
        prog.step("mix")(0.0, i18n(STEP_TEXT["mix"]))
        result.folder, result.cover, result.stems = self._mix(song, voice, options, instrumental, converted_lead,
                                                              backing, exports, lead_label)
        log.done()
        log.add(i18n("Cover finished in {seconds} s. Saved in: {folder}").format(
            seconds=round(time.monotonic() - started), folder=result.folder))
        progress(1.0, i18n("Cover finished"))
        return result

    def _mix(self, song: Path, voice: Voice, options: CoverOptions, instrumental: Path, lead: Path,
             backing: Path | None, exports: dict[str, Path], lead_label: str) -> tuple[Path, Path, dict[str, Path]]:
        """Mix the cover into ``outputs/<song> (<voice> Cover)/`` and save every stem next to it."""
        base = _output_name(song, voice)
        folder = self.outputs_dir / base
        _clear_previous_outputs(folder, song)
        cover = folder / f"{base}.{options.output_format}"
        with tempfile.TemporaryDirectory(prefix="mix-", dir=self._tmp_root()) as tmp:
            import soundfile as sf

            processed_lead = lead
            if options.effects.enabled:
                lead_sr = sf.info(str(lead)).samplerate
                processed_lead = write_audio(Path(tmp) / "lead.wav",
                                             apply_effects(load_audio(lead, lead_sr), lead_sr, options.effects), lead_sr)
                exports = {**exports, f"{lead_label}, Effects": processed_lead}
            tracks = [Track(instrumental, BASE_LEVELS_DB["instrumental"] + options.instrumental_gain_db),
                      Track(processed_lead, BASE_LEVELS_DB["lead"] + options.lead_gain_db)]
            if backing is not None:
                tracks.append(Track(backing, BASE_LEVELS_DB["backing"] + options.backing_gain_db))
            mix_tracks(tracks, cover)
            saved = {path: _export_stem(path, folder / _stem_name(song, label, options.output_format))
                     for label, path in exports.items()}
        stems = {"instrumental": saved[instrumental], "lead": saved[processed_lead]}
        if backing is not None:
            stems["backing"] = saved[backing]
        return folder, cover, stems

    def _tmp_root(self) -> Path:
        root = Path(self.workspace_root or paths.WORKSPACE_DIR) / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def run_batch(self, input_dir: Path, options: CoverOptions, progress: ProgressFn = _noop) -> Iterator[str]:
        """Make a cover of every audio file in ``input_dir``. Yields log lines; failures are skipped."""
        files = list_audio_files(Path(input_dir))
        if not files:
            yield "No valid audio files."
            return
        yield f"{len(files)} audio files found"
        failed = []
        for i, file in enumerate(files):
            yield f"Making the cover of {file.name}..."
            step = lambda fraction, desc, i=i: progress((i + fraction) / len(files), f"{i + 1}/{len(files)}: {desc}")
            try:
                result = self.run(file, options, step)
                yield f"Done: {result.folder.name}"
            except Exception as e:
                logger.warning("Cover of %s failed", file, exc_info=not isinstance(e, AppError))
                failed.append(file.name)
                yield f"Error with {file.name}: {e}"
        yield f"Finished: {len(files) - len(failed)} covers, {len(failed)} failed. Output: {self.outputs_dir}"


def _sum_files(files: list[Path], dest: Path) -> Path:
    """One file with the sum of ``files`` (e.g. drums + bass + other = instrumental)."""
    if len(files) == 1:
        return files[0]
    import soundfile as sf

    sample_rate = sf.info(str(files[0])).samplerate
    waves = [load_audio(f, sample_rate) for f in files]
    length = max(w.shape[1] for w in waves)
    channels = max(w.shape[0] for w in waves)
    total = np.zeros((channels, length), dtype=np.float32)
    for w in waves:
        total[: w.shape[0], : w.shape[1]] += w
    return write_audio(dest, total, sample_rate)


def _export_stem(source: Path, dest: Path) -> Path:
    """Copy a stem to the output folder in the cover's format."""
    if source.suffix.lower() == dest.suffix.lower():
        return Path(shutil.copy2(source, dest))
    import soundfile as sf

    sample_rate = sf.info(str(source)).samplerate
    return write_audio(dest, load_audio(source, sample_rate), sample_rate)


def _clear_previous_outputs(folder: Path, song: Path) -> None:
    """Remove the audio files a previous run of this cover left, so no stale stem is mixed up with the new ones."""
    folder.mkdir(parents=True, exist_ok=True)
    prefix = _safe_file_name(f"{song.stem} (")
    for entry in folder.iterdir():
        if entry.is_file() and entry.name.startswith(prefix) and entry.suffix.lower() in AUDIO_EXTENSIONS:
            entry.unlink()


def _resolve_voice(voice: str | Voice) -> Voice:
    if isinstance(voice, Voice):
        return voice
    if not voice:
        raise AppError("Please select a voice model")
    try:
        return get_voice(voice)
    except KeyError as e:
        raise AppError(f"Voice model not found: {voice}") from e


def _display_name(voice: Voice) -> str:
    return voice.name.replace("/", " ")


def _safe_file_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")


def _stem_name(song: Path, label: str, output_format: str) -> str:
    return _safe_file_name(f"{song.stem} ({label})") + f".{output_format}"


def _output_name(song: Path, voice: Voice) -> str:
    return _safe_file_name(f"{song.stem} ({_display_name(voice)} Cover)")
