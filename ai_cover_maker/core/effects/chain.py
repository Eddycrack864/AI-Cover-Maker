"""Audio effects with Spotify's pedalboard. Waves are float32 ``(channels, samples)``."""

from dataclasses import dataclass

import numpy as np

MAX_SEMITONES = 24


@dataclass(frozen=True)
class VocalEffects:
    """Post-processing of the converted lead vocals. Only the checked effects are applied.

    The defaults are AICoverGen's chain, which it always applies: a 50 Hz high-pass, a gentle
    compressor and a small, mostly dry room.
    """

    highpass: bool = True
    highpass_hz: float = 50.0
    compressor: bool = True
    compressor_threshold_db: float = -15.0
    compressor_ratio: float = 4.0
    reverb: bool = True
    reverb_room_size: float = 0.15
    reverb_damping: float = 0.7
    reverb_wet_level: float = 0.2
    reverb_dry_level: float = 0.8
    reverb_width: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.highpass or self.compressor or self.reverb

    def board(self):
        from pedalboard import Compressor, HighpassFilter, Pedalboard, Reverb

        plugins = []
        if self.highpass:
            plugins.append(HighpassFilter(cutoff_frequency_hz=self.highpass_hz))
        if self.compressor:
            plugins.append(Compressor(threshold_db=self.compressor_threshold_db, ratio=self.compressor_ratio))
        if self.reverb:
            plugins.append(Reverb(room_size=self.reverb_room_size, damping=self.reverb_damping,
                                  wet_level=self.reverb_wet_level, dry_level=self.reverb_dry_level,
                                  width=self.reverb_width))
        return Pedalboard(plugins)


def _as_float32(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    return wave[np.newaxis, :] if wave.ndim == 1 else wave


def apply_effects(wave: np.ndarray, sample_rate: int, effects: VocalEffects) -> np.ndarray:
    wave = _as_float32(wave)
    if not effects.enabled:
        return wave
    if effects.reverb and wave.shape[0] == 1:
        wave = np.repeat(wave, 2, axis=0)  # stereo, or the reverb width has nothing to spread
    return effects.board()(wave, sample_rate)


def pitch_shift(wave: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Change the key keeping the duration (unlike resampling, which also changes the tempo)."""
    wave = _as_float32(wave)
    if not semitones:
        return wave
    if abs(semitones) > MAX_SEMITONES:
        raise ValueError(f"Pitch shift must be between -{MAX_SEMITONES} and {MAX_SEMITONES} semitones")
    from pedalboard import PitchShift

    return PitchShift(semitones=float(semitones))(wave, sample_rate)
