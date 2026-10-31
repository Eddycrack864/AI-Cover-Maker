import numpy as np
import pytest
import soundfile as sf

from ai_cover_maker.core.effects.chain import VocalEffects, apply_effects, pitch_shift
from ai_cover_maker.core.errors import MixError
from ai_cover_maker.core.mix.mixer import Track, db_to_gain, limit, mix_tracks, mix_waves

SR = 44100


def sine(freq=440.0, seconds=1.0, sr=SR, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(wave, sr=SR):
    spectrum = np.abs(np.fft.rfft(wave))
    return np.fft.rfftfreq(len(wave), 1 / sr)[spectrum.argmax()]


def test_mix_waves_pads_and_upmixes():
    mono = sine(seconds=1.0)[np.newaxis]
    stereo = np.stack([sine(seconds=0.5)] * 2)
    mixed = mix_waves([mono, stereo], [0.0, -6.0])
    assert mixed.shape == (2, SR)
    np.testing.assert_allclose(mixed[:, SR // 2 + 10], mono[0, SR // 2 + 10], atol=1e-6)
    with pytest.raises(MixError):
        mix_waves([], [])


def test_limit_keeps_peaks_below_the_ceiling():
    loud = np.stack([sine(amp=1.0)] * 2) * 3
    limited = limit(loud, SR, ceiling_db=-1.0)
    assert np.abs(limited).max() <= db_to_gain(-1.0) + 1e-6


def test_mix_tracks_resamples_and_writes(tmp_path):
    inst = tmp_path / "inst.wav"
    vocals = tmp_path / "vocals.wav"
    sf.write(inst, np.stack([sine(220, 1.0)] * 2).T, SR)
    sf.write(vocals, sine(440, 0.5, sr=40000, amp=0.9), 40000)  # RVC output: mono, model sample rate
    out = mix_tracks([Track(inst, -3), Track(vocals, 0)], tmp_path / "cover.flac")
    data, sr = sf.read(out, always_2d=True)
    assert sr == SR and data.shape == (SR, 2)
    assert np.abs(data).max() <= db_to_gain(-1.0) + 1e-3
    with pytest.raises(MixError):
        mix_tracks([Track(tmp_path / "missing.wav")], tmp_path / "x.wav")


def test_pitch_shift_keeps_duration_and_changes_key():
    wave = sine(440, 1.0)
    shifted = pitch_shift(wave, SR, 12)
    assert shifted.shape == (1, len(wave))  # same length: the tempo doesn't change
    assert abs(dominant_freq(shifted[0, SR // 4:]) - 880) < 20
    assert pitch_shift(wave, SR, 0).shape == (1, len(wave))
    with pytest.raises(ValueError):
        pitch_shift(wave, SR, 30)


def test_vocal_effects():
    wave = sine(seconds=0.5)
    assert apply_effects(wave, SR, VocalEffects()).shape == (1, len(wave))  # all off: untouched
    wet = apply_effects(wave, SR, VocalEffects(reverb=True, highpass=True, compressor=True))
    assert wet.shape[0] == 2 and np.isfinite(wet).all()
