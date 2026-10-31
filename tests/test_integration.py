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


def test_cover_pipeline_end_to_end(sample, voice, tmp_path):
    """Real separation + RVC on the sample, then a second run that only re-mixes."""
    from dataclasses import replace

    from ai_cover_maker.core.pipeline.options import CoverOptions
    from ai_cover_maker.core.pipeline.runner import CoverPipeline

    pipeline = CoverPipeline(workspace_root=tmp_path / "workspace", outputs_dir=tmp_path / "outputs")
    options = CoverOptions(voice=voice.name, vocals_model="MelBand Roformer | Vocals by Kimberley Jensen")
    result = pipeline.run(sample, options)
    data, sr = sf.read(result.cover)
    assert data.ndim == 2 and np.abs(data).max() <= 10 ** (-1 / 20) + 1e-3
    assert abs(len(data) / sr - sf.info(sample).duration) < 0.5

    remix = pipeline.run(sample, replace(options, instrumental_gain_db=-6, output_format="mp3"))
    assert remix.cached_steps == ["vocals", "rvc_lead"] and remix.cover.suffix == ".mp3"
