"""Mix stems into the final cover."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_cover_maker.core.audio import load_audio, write_audio
from ai_cover_maker.core.errors import MixError

DEFAULT_CEILING_DB = -1.0
# Level of each track before the user's gain, as in AICoverGen: the lead vocals sit 3 dB above the
# instrumental, and the sum has enough headroom that the limiter only catches the odd peak instead
# of squashing the whole mix.
BASE_LEVELS_DB = {"lead": -4.0, "backing": -6.0, "instrumental": -7.0}


@dataclass(frozen=True)
class Track:
    path: Path
    gain_db: float = 0.0


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def to_stereo(wave: np.ndarray) -> np.ndarray:
    if wave.shape[0] == 1:
        return np.repeat(wave, 2, axis=0)
    return wave[:2]


def mix_waves(waves: list[np.ndarray], gains_db: list[float]) -> np.ndarray:
    """Sum ``(channels, samples)`` waves at the same sample rate; shorter ones are padded with silence."""
    if not waves:
        raise MixError("Nothing to mix")
    length = max(w.shape[1] for w in waves)
    mixed = np.zeros((2, length), dtype=np.float32)
    for wave, gain_db in zip(waves, gains_db):
        stereo = to_stereo(np.asarray(wave, dtype=np.float32))
        mixed[:, :stereo.shape[1]] += stereo * db_to_gain(gain_db)
    return mixed


def limit(wave: np.ndarray, sample_rate: int, ceiling_db: float = DEFAULT_CEILING_DB) -> np.ndarray:
    """Limiter that keeps the peaks at or below ``ceiling_db`` dBFS."""
    from pedalboard import Limiter

    limited = Limiter(threshold_db=ceiling_db, release_ms=100.0)(np.asarray(wave, dtype=np.float32), sample_rate)
    # The limiter has a short attack: catch the transients it lets through
    ceiling = db_to_gain(ceiling_db)
    peak = float(np.abs(limited).max()) if limited.size else 0.0
    return limited * (ceiling / peak) if peak > ceiling else limited


def mix_tracks(tracks: list[Track], output_path: Path, *, sample_rate: int | None = None,
               limiter: bool = True, ceiling_db: float = DEFAULT_CEILING_DB) -> Path:
    """Mix audio files into ``output_path`` (format from its extension).

    The sample rate defaults to the first track's, so put the instrumental first.
    """
    tracks = [t for t in tracks if t.path]
    if not tracks:
        raise MixError("Nothing to mix")
    if missing := [str(t.path) for t in tracks if not Path(t.path).is_file()]:
        raise MixError(f"Audio not found: {', '.join(missing)}")
    try:
        import soundfile as sf

        if sample_rate is None:
            try:
                sample_rate = sf.info(str(tracks[0].path)).samplerate
            except Exception:
                sample_rate = 44100
        waves = [load_audio(t.path, sample_rate) for t in tracks]
        mixed = mix_waves(waves, [t.gain_db for t in tracks])
        if limiter:
            mixed = limit(mixed, sample_rate, ceiling_db)
        return write_audio(output_path, mixed, sample_rate)
    except MixError:
        raise
    except Exception as e:
        raise MixError(f"Mixing failed: {str(e) or type(e).__name__}") from e
