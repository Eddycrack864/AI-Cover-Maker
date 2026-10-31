"""End-to-end checks with real models. Run with ``pytest --slow``.

They use local, gitignored test assets: a short song in ``.local/samples/`` and any voice
model in ``models/rvc/``. Missing assets skip the test; missing weights are downloaded.
"""

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.slow

SAMPLES = sorted((REPO_ROOT / ".local" / "samples").glob("*.wav")) if (REPO_ROOT / ".local" / "samples").is_dir() else []


@pytest.fixture(scope="module")
def sample():
    if not SAMPLES:
        pytest.skip("no test song in .local/samples/")
    return SAMPLES[0]


@pytest.fixture(scope="module")
def voice():
    from ai_cover_maker.core.rvc.voices import list_voices

    voices = list_voices()
    if not voices:
        pytest.skip("no voice model in models/rvc/")
    return voices[-1]


def test_rvc_conversion(sample, voice, tmp_path):
    from ai_cover_maker.core.rvc.engine import RvcEngine, RvcParams

    engine = RvcEngine()
    out = engine.convert(sample, tmp_path / "converted.wav", voice, RvcParams(pitch=2, f0_autotune=True))
    data, sr = sf.read(out)
    assert sr in (32000, 40000, 48000) and np.isfinite(data).all() and np.abs(data).max() > 0.01
    assert abs(len(data) / sr - sf.info(sample).duration) < 0.5
    engine.unload()
