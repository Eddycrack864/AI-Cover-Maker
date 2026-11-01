import zipfile
from pathlib import Path

import pytest

from ai_cover_maker.core.errors import ConversionError, DownloadError
from ai_cover_maker.core.rvc import assets
from ai_cover_maker.core.rvc.engine import RvcEngine, RvcParams
from ai_cover_maker.core.rvc.voices import (
    Voice, delete_voice, get_voice, install_voice, list_voices, safe_folder_name,
)


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def voices_dir(tmp_path):
    root = tmp_path / "rvc"
    touch(root / "Atri" / "atri_e150.pth")
    touch(root / "Atri" / "atri_e300.pth")
    touch(root / "Atri" / "trained_IVF_atri.index")
    touch(root / "Atri" / "added_IVF_atri.index")
    touch(root / "Solo" / "solo.pth")
    touch(root / "Solo" / "G_2333.pth")  # training checkpoint
    touch(root / "NoIndex" / "voice.onnx")
    touch(root / "zips" / "pending.pth")  # Applio's download folder
    touch(root / "Empty" / "readme.txt")
    return root


def test_list_voices(voices_dir):
    voices = {v.name: v for v in list_voices(voices_dir)}
    assert sorted(voices) == ["Atri/atri_e150", "Atri/atri_e300", "NoIndex", "Solo"]
    assert voices["Atri/atri_e300"].index_path.name == "added_IVF_atri.index"
    assert voices["Solo"].model_path.name == "solo.pth" and voices["Solo"].index_path is None
    assert get_voice("NoIndex", voices_dir).model_path.suffix == ".onnx"
    with pytest.raises(KeyError):
        get_voice("Missing", voices_dir)
    assert list_voices(voices_dir / "nowhere") == []


def test_install_voice_from_files_and_zip(tmp_path):
    voices_dir = tmp_path / "rvc"
    pth = touch(tmp_path / "in" / "Singer.pth")
    index = touch(tmp_path / "in" / "added_Singer.index")
    folder = install_voice([pth, index], "My Singer", voices_dir)
    assert folder == voices_dir / "My_Singer"
    assert {p.name for p in folder.iterdir()} == {"Singer.pth", "added_Singer.index"}

    with pytest.raises(DownloadError, match="already exists"):
        install_voice([pth], "My Singer", voices_dir)

    archive = tmp_path / "in" / "Other.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Other/model.pth", b"x")
        zf.writestr("Other/added.index", b"x")
        zf.writestr("__MACOSX/Other/._model.pth", b"x")
    folder = install_voice([archive], "Other", voices_dir)
    assert {p.name for p in folder.iterdir()} == {"model.pth", "added.index"}


def test_install_voice_rejects_bad_input(tmp_path):
    voices_dir = tmp_path / "rvc"
    with pytest.raises(DownloadError, match="Unsupported"):
        install_voice([touch(tmp_path / "a.txt")], "A", voices_dir)
    with pytest.raises(DownloadError, match="No voice model"):
        install_voice([touch(tmp_path / "a.index")], "A", voices_dir)
    with pytest.raises(DownloadError, match="name"):
        install_voice([touch(tmp_path / "a.pth")], " / ", voices_dir)

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escape.pth", b"x")
    with pytest.raises(DownloadError, match="unsafe"):
        install_voice([evil], "Evil", voices_dir)
    assert not (tmp_path / "escape.pth").exists()


def test_safe_folder_name():
    assert safe_folder_name(' My: "Voice" v2 ') == "My___Voice__v2"
    assert safe_folder_name("../..") == ""


def test_delete_voice(voices_dir):
    # One checkpoint of a folder: the others keep the shared index
    delete_voice("Atri/atri_e150", voices_dir)
    assert sorted(p.name for p in (voices_dir / "Atri").iterdir()) == [
        "added_IVF_atri.index", "atri_e300.pth", "trained_IVF_atri.index"]
    # The last checkpoint takes the folder with it
    delete_voice("Atri/atri_e300", voices_dir)
    assert not (voices_dir / "Atri").exists()
    delete_voice("Solo", voices_dir)
    assert not (voices_dir / "Solo").exists()
    for bad in ("..", "zips", "Missing", "NoIndex/missing"):
        with pytest.raises(DownloadError):
            delete_voice(bad, voices_dir)


def test_required_assets(tmp_path, monkeypatch):
    from ai_cover_maker import paths

    monkeypatch.setattr(paths, "RVC_ASSETS_DIR", tmp_path)
    files = assets.required_files("rmvpe", "contentvec")
    assert sorted(p.relative_to(tmp_path).as_posix() for p in files.values()) == [
        "embedders/contentvec/config.json", "embedders/contentvec/pytorch_model.bin", "predictors/rmvpe.pt"]
    assert all(url.startswith(assets.HF_RESOURCES) for url in files)
    assert list(assets.required_files("crepe", "custom")) == []
    touch(tmp_path / "predictors" / "rmvpe.pt")
    assert len(assets.missing_files("rmvpe", "contentvec")) == 2


class FakeConverter:
    calls = []
    fail = False

    def convert_audio(self, **kwargs):
        if FakeConverter.fail:
            raise RuntimeError("CUDA out of memory")
        FakeConverter.calls.append(kwargs)
        Path(kwargs["audio_output_path"]).write_bytes(b"RIFF")

    def cleanup_model(self):
        pass


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(assets, "ensure_assets", lambda *a, **k: None)
    FakeConverter.calls.clear()
    FakeConverter.fail = False
    return RvcEngine(factory=FakeConverter)


def test_engine_convert(engine, tmp_path):
    voice = Voice("Solo", touch(tmp_path / "solo.pth"), None)
    source = touch(tmp_path / "vocals.wav")
    out = engine.convert(source, tmp_path / "out" / "converted.wav", voice, RvcParams(pitch=3, f0_autotune=True))
    assert out.read_bytes() == b"RIFF"
    call = FakeConverter.calls[0]
    assert call["pitch"] == 3 and call["f0_autotune"] is True  # a real bool, not the string "True"
    assert call["index_path"] == "" and call["export_format"] == "WAV"
    assert not list((tmp_path / "out").glob(".*part*"))


def test_engine_errors(engine, tmp_path):
    voice = Voice("Solo", touch(tmp_path / "solo.pth"), None)
    source = touch(tmp_path / "vocals.wav")
    with pytest.raises(ConversionError, match="Input audio not found"):
        engine.convert(tmp_path / "missing.wav", tmp_path / "o.wav", voice)
    with pytest.raises(ConversionError, match="Voice model not found"):
        engine.convert(source, tmp_path / "o.wav", Voice("X", tmp_path / "x.pth"))
    with pytest.raises(ConversionError, match="pitch extraction"):
        engine.convert(source, tmp_path / "o.wav", voice, RvcParams(f0_method="harvest"))
    FakeConverter.fail = True
    with pytest.raises(ConversionError, match="out of memory"):
        engine.convert(source, tmp_path / "o.wav", voice)
    assert not (tmp_path / "o.wav").exists()
