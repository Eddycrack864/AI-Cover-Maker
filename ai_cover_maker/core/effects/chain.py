"""Audio effects with Spotify's pedalboard. Waves are float32 ``(channels, samples)``."""

from dataclasses import dataclass

import numpy as np

MAX_SEMITONES = 24


@dataclass(frozen=True)
class VocalEffects:
    """Post-processing of the converted vocals. Every effect is off by default."""

    highpass: bool = False
    highpass_hz: float = 80.0
    compressor: bool = False
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 3.0
    reverb: bool = False
    reverb_room_size: float = 0.5
    reverb_damping: float = 0.5
    reverb_wet_level: float = 0.33
    reverb_dry_level: float = 0.4
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
