import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_cover_maker.core.errors import AppError, SeparationError
from ai_cover_maker.core.models.catalog import ModelCatalog
from ai_cover_maker.core.pipeline.cache import SongWorkspace, clear_workspace, stage_key
from ai_cover_maker.core.pipeline.options import CoverOptions, load_presets, preset_options
from ai_cover_maker.core.pipeline.roles import split_stems
from ai_cover_maker.core.pipeline.runner import CoverPipeline
from ai_cover_maker.core.rvc.engine import RvcParams
from ai_cover_maker.core.rvc.voices import Voice
from ai_cover_maker.core.separation import runtime
from ai_cover_maker.core.separation.runtime import SeparatorCache
from tests.conftest import REPO_ROOT

SR = 44100
VOCALS = "MelBand Roformer | Vocals by Kimberley Jensen"
SIX_STEMS = "BS Roformer SW by jarredou"
KARAOKE = "Mel-Roformer-Karaoke-Aufr33-Viperx"
DEREVERB = "MelBand Roformer | De-Reverb by anvuew"
DEREVERB_VR = "UVR-DeEcho-DeReverb.pth"

# model filename -> labels audio-separator puts in the output names
STEMS = {
    "vocals_mel_band_roformer.ckpt": ["vocals", "other"],
    "BS-Roformer-SW.ckpt": ["bass", "drums", "guitar", "other", "piano", "vocals"],
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": ["Vocals", "Instrumental"],
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": ["noreverb", "reverb"],
    "UVR-DeEcho-DeReverb.pth": ["No Reverb", "Reverb"],
}


class WavSeparator:
    """Writes a short sine per stem, like audio-separator would write the separated audio."""

    runs = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_model(self, model_filename):
        self.model = model_filename

    def separate(self, path):
        sf.info(path)  # unreadable audio fails, as in audio-separator
        WavSeparator.runs.append((self.model, Path(path).name))
        names = []
        for i, label in enumerate(STEMS[self.model]):
            name = f"{Path(path).stem}_({label})_{Path(self.model).stem}.wav"
            t = np.arange(SR // 2) / SR
            sf.write(Path(self.kwargs["output_dir"]) / name, np.stack([0.1 * np.sin(2 * np.pi * (200 + 50 * i) * t)] * 2).T, SR)
            names.append(name)
        return names


class FakeEngine:
    def __init__(self):
        self.calls = []

    def convert(self, input_path, output_path, voice, params, progress=None):
        self.calls.append((voice.name, params.pitch, Path(input_path).parent.name))
        data, sr = sf.read(input_path)
        sf.write(output_path, data.mean(axis=1) if data.ndim == 2 else data, 40000 if sr else sr)
        return Path(output_path)

    def unload(self):
        pass


@pytest.fixture
def catalog():
    return ModelCatalog.from_file(REPO_ROOT / "assets" / "models.json")


@pytest.fixture
def pipeline(tmp_path, catalog, monkeypatch):
    monkeypatch.setattr(runtime, "use_autocast", lambda: False)
    monkeypatch.setattr(runtime, "ensure_downloaded", lambda *a, **k: None)
    # The 6-stem test model is the SW entry of the catalog, renamed to a short file name
    sw = catalog.find(SIX_STEMS)
    STEMS[sw.filename] = STEMS.pop("BS-Roformer-SW.ckpt", STEMS.get(sw.filename))
    WavSeparator.runs.clear()
    defaults = json.loads((REPO_ROOT / "assets" / "default_settings.json").read_text(encoding="utf-8"))
    return CoverPipeline(catalog=catalog, separation_settings=defaults, separator=SeparatorCache(factory=WavSeparator),
                         engine=FakeEngine(), workspace_root=tmp_path / "workspace", outputs_dir=tmp_path / "outputs")


@pytest.fixture
def song(tmp_path):
    path = tmp_path / "My Song.wav"
    sf.write(path, np.zeros((SR, 2), dtype=np.float32), SR)
    return path


@pytest.fixture
def voice(tmp_path):
    model = tmp_path / "voices" / "Singer" / "singer.pth"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    return Voice("Singer", model)


def test_full_cover_and_cache(pipeline, song, voice):
    options = CoverOptions(voice=voice, vocals_model=VOCALS, karaoke_model=KARAOKE, dereverb_model=DEREVERB)
    result = pipeline.run(song, options)

    assert result.cover.name == "My Song (Singer Cover).wav"
    data, sr = sf.read(result.cover)
    assert sr == SR and data.ndim == 2 and np.abs(data).max() <= 10 ** (-1 / 20) + 1e-3
    assert [run[0] for run in WavSeparator.runs] == [
        "vocals_mel_band_roformer.ckpt", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"]
    assert pipeline.engine.calls == [("Singer", 0, pipeline.engine.calls[0][2])]
    assert set(result.stems) == {"instrumental", "backing"}
    assert result.cached_steps == []

    # Only the mix changed: nothing is separated or converted again
    WavSeparator.runs.clear()
    again = pipeline.run(song, replace(options, lead_gain_db=-2, output_format="flac"))
    assert WavSeparator.runs == [] and len(pipeline.engine.calls) == 1
    assert again.cached_steps == ["vocals", "karaoke", "dereverb", "rvc_lead"]
    assert again.cover.suffix == ".flac"


def test_changing_a_step_recomputes_only_what_depends_on_it(pipeline, song, voice):
    options = CoverOptions(voice=voice, vocals_model=VOCALS, dereverb_model=DEREVERB)
    pipeline.run(song, options)
    WavSeparator.runs.clear()

    result = pipeline.run(song, replace(options, dereverb_model=DEREVERB_VR))
    assert [run[0] for run in WavSeparator.runs] == ["UVR-DeEcho-DeReverb.pth"]
    assert result.cached_steps == ["vocals"]
    assert len(pipeline.engine.calls) == 2  # the lead changed, so it is converted again

    result = pipeline.run(song, replace(options, dereverb_model=DEREVERB_VR, rvc=RvcParams(index_rate=0.9)))
    assert result.cached_steps == ["vocals", "dereverb"] and len(pipeline.engine.calls) == 3


def test_key_shift_moves_rvc_pitch_and_instrumental(pipeline, song, voice):
    options = CoverOptions(voice=voice, vocals_model=VOCALS, karaoke_model=KARAOKE, key_shift=-2,
                           rvc=RvcParams(pitch=12))
    result = pipeline.run(song, options)
    assert pipeline.engine.calls[0][1] == 10  # voice pitch + key shift
    assert any("shift_instrumental" in line or "key changed" in line for line in result.log)
    inst, _ = sf.read(result.stems["instrumental"])
    assert len(inst) == SR // 2  # same duration after the key change
    assert "shift_backing" in str(result.stems["backing"].parent.name)


def test_backing_vocals_conversion(pipeline, song, voice):
    options = CoverOptions(voice=voice, vocals_model=VOCALS, karaoke_model=KARAOKE, backing_voice=voice,
                           backing_rvc=RvcParams(pitch=-3))
    result = pipeline.run(song, options)
    assert [c[1] for c in pipeline.engine.calls] == [0, -3]
    assert result.stems["backing"].parent.name.startswith("rvc_backing-")
    with pytest.raises(AppError, match="lead/backing"):
        pipeline.run(song, replace(options, karaoke_model=None))


def test_multi_stem_model_sums_the_instrumental(pipeline, song, voice):
    result = pipeline.run(song, CoverOptions(voice=voice, vocals_model=SIX_STEMS, keep_stems=True))
    inst, _ = sf.read(result.stems["instrumental"])
    assert inst.shape == (SR // 2, 2) and np.abs(inst).max() > 0.2  # five sines summed
    assert result.stems["lead"].parent.name.endswith("(stems)")


def test_errors(pipeline, song, voice, tmp_path):
    with pytest.raises(AppError, match="not found"):
        pipeline.run(tmp_path / "missing.wav", CoverOptions(voice=voice, vocals_model=VOCALS))
    with pytest.raises(SeparationError, match="Unknown separation model"):
        pipeline.run(song, CoverOptions(voice=voice, vocals_model="Nope"))
    with pytest.raises(AppError, match="Voice model not found"):
        pipeline.run(song, CoverOptions(voice="Nobody", vocals_model=VOCALS))


def test_batch_reports_each_file(pipeline, song, voice, tmp_path):
    (tmp_path / "broken.wav").write_bytes(b"not audio")
    lines = list(pipeline.run_batch(tmp_path, CoverOptions(voice=voice, vocals_model=VOCALS)))
    assert lines[0] == "2 audio files found"
    assert lines[-1].startswith("Finished: 1 covers, 1 failed")


def test_split_stems():
    assert split_stems("dereverb", {"No Reverb": "a", "Reverb": "b"}) == ("a", ["b"])
    assert split_stems("denoise", {"noise": "a", "No Noise": "b"}) == ("b", ["a"])
    assert split_stems("dereverb", {"dry": "a", "no dry": "b"}) == ("a", ["b"])
    assert split_stems("vocals", {"bass": "a", "Vocals": "b", "other": "c"}) == ("b", ["a", "c"])
    with pytest.raises(SeparationError, match="Pick another model"):
        split_stems("denoise", {"left": "a", "right": "b"})


def test_workspace(tmp_path, song):
    ws = SongWorkspace(song, tmp_path / "ws")
    assert ws.load("vocals", "k") is None
    src = tmp_path / "x.wav"
    src.write_bytes(b"RIFF")
    stored = ws.store("vocals", "k", {"kept": src})
    assert ws.load("vocals", "k") == stored and not src.exists()
    stored["kept"].unlink()
    assert ws.load("vocals", "k") is None  # incomplete stages are not reused
    assert stage_key("a", {"x": 1}) != stage_key("a", {"x": 2}) and len(stage_key("a", 1)) == 16
    assert clear_workspace(tmp_path / "ws") == 1


def test_presets_use_catalog_models(catalog):
    presets = load_presets()
    assert list(presets) == ["Balanced", "Fast", "Best quality"]
    for name, preset in presets.items():
        options = preset_options(preset)
        assert options["vocals_model"], name
        for stage in ("vocals", "karaoke", "dereverb", "deecho", "denoise"):
            model_id = options[f"{stage}_model"]
            assert model_id is None or catalog.find(model_id) is not None, (name, stage, model_id)
