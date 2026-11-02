import json
import re

import pytest

from ai_cover_maker.core.models.catalog import Model, ModelCatalog, load_recommended
from tests.conftest import REPO_ROOT


@pytest.fixture(scope="module")
def catalog():
    return ModelCatalog.from_file(REPO_ROOT / "assets" / "models.json")


def test_counts_per_arch(catalog):
    counts = {arch: len(catalog.by_arch(arch)) for arch in ("roformer", "mdx23c", "mdxnet", "vrarch", "demucs")}
    assert counts == {"roformer": 92, "mdx23c": 5, "mdxnet": 39, "vrarch": 29, "demucs": 4}
    assert len(catalog) == sum(counts.values())


def test_dropdown_order_matches_legacy(catalog):
    roformer = catalog.ids("roformer")
    assert roformer[:2] == ["BS-Roformer-Viperx-1297", "BS-Roformer-Viperx-1296"]
    assert roformer[-1] == "BS Roformer SW by jarredou"
    assert catalog.ids("mdx23c")[0] == "MDX23C_D1581.ckpt"
    assert catalog.ids("mdxnet")[-1] == "kuielab_b_drums.onnx"
    assert catalog.ids("vrarch")[0] == "1_HP-UVR.pth"
    assert catalog.ids("demucs") == ["htdemucs_ft.yaml", "htdemucs_6s.yaml", "htdemucs.yaml", "hdemucs_mmi.yaml"]


def test_ids_unique_and_filenames_present(catalog):
    for arch in ("roformer", "mdx23c", "mdxnet", "vrarch", "demucs"):
        ids = catalog.ids(arch)
        assert len(ids) == len(set(ids))
    assert all(m.filename for m in catalog)


def test_roformer_display_name_maps_to_file(catalog):
    model = catalog.get("roformer", "BS Roformer SW by jarredou")
    assert model.filename == "BS-Roformer-SW.ckpt"
    with pytest.raises(KeyError):
        catalog.get("mdxnet", "BS Roformer SW by jarredou")


def test_filenames_unique_and_lookup(catalog):
    filenames = [m.filename for m in catalog]
    assert len(filenames) == len(set(filenames))
    assert catalog.by_filename("BS-Roformer-SW.ckpt").id == "BS Roformer SW by jarredou"
    assert catalog.by_filename("missing.ckpt") is None


def test_downloadable_excludes_models_without_urls(catalog):
    downloadable = {m.id for m in catalog.downloadable()}
    assert "htdemucs_ft.yaml" not in downloadable  # Demucs weights are not on GitHub nor the mirror
    assert "UVR-MDX-NET-Inst_HQ_3.onnx" in downloadable


def test_urls_source_selection_and_fallback():
    both = Model("m", "mdxnet", "m.onnx", {"github": ["https://github.com/x"], "huggingface": ["https://huggingface.co/x"]})
    assert both.urls("huggingface") == ["https://huggingface.co/x"]
    assert both.urls("github") == ["https://github.com/x"]
    only_gh = Model("m", "mdxnet", "m.onnx", {"github": ["https://github.com/x"]})
    assert only_gh.urls("huggingface") == ["https://github.com/x"]
    assert Model("m", "demucs", "m.yaml").urls("github") == []


def test_downloaded(catalog, tmp_path):
    (tmp_path / "UVR-MDX-NET-Inst_HQ_3.onnx").touch()
    (tmp_path / "BS-Roformer-SW.ckpt").touch()
    assert [m.id for m in catalog.downloaded("mdxnet", tmp_path)] == ["UVR-MDX-NET-Inst_HQ_3.onnx"]
    assert [m.id for m in catalog.downloaded("roformer", tmp_path)] == ["BS Roformer SW by jarredou"]
    assert catalog.downloaded("vrarch", tmp_path / "missing") == []


def test_catalog_has_audio_separator_own_models(catalog):
    """Every model of audio-separator's bundled list (the ones it adds to UVR's) is in the catalog.

    UVR's own list is downloaded at runtime, so it's checked by tests/test_model_links.py (--network).
    """
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.find_spec("audio_separator")
    if spec is None:
        pytest.skip("audio-separator is not installed")
    bundled = json.loads((Path(next(iter(spec.submodule_search_locations))) / "models.json").read_text(encoding="utf-8"))
    filenames = set()
    for models in bundled.values():
        for value in models.values():
            filenames.add(next(iter(value)) if isinstance(value, dict) else value)
    missing = sorted(filenames - {m.filename for m in catalog})
    assert not missing, f"Run tools/sync_models.py --write to add: {missing}"


RECOMMENDED_FILE = REPO_ROOT / "assets" / "recommended_models.json"


def test_recommended_models_are_in_the_catalog(catalog):
    recommended = load_recommended(RECOMMENDED_FILE)
    assert recommended, "assets/recommended_models.json is empty or invalid"
    missing = {stem: model_id for stem, model_id in recommended.items() if catalog.find(model_id) is None}
    assert not missing, f"Recommended models not found in assets/models.json (use the exact id): {missing}"


def test_recommended_first_and_stems(tmp_path):
    models = [Model(name, "roformer", f"{name}.ckpt") for name in ("a", "b", "c", "d")]
    catalog = ModelCatalog(models, {"Vocals": "c", "Instrumental": "a", "Karaoke": "c", "Gone": "zzz"})
    assert [m.id for m in catalog.recommended_first(models)] == ["c", "a", "b", "d"]
    assert catalog.recommended_stems(models[2]) == ["Vocals", "Karaoke"]
    assert catalog.recommended_stems(models[1]) == []
    assert catalog.recommended_first(models[1:2]) == models[1:2]


def test_recommended_file_is_optional(tmp_path):
    assert load_recommended(tmp_path / "missing.json") == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert load_recommended(tmp_path / "bad.json") == {}
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 2, "models": [
        {"id": "a", "arch": "roformer", "filename": "a.ckpt"}]}))
    (tmp_path / "recommended_models.json").write_text(json.dumps({"version": 1, "recommended": {"Vocals": "a"}}))
    catalog = ModelCatalog.from_file(tmp_path / "catalog.json")
    assert catalog.recommended_stems(catalog.find("a")) == ["Vocals"]



# recommended_models.json stem -> the cover step it is recommended for
RECOMMENDED_STAGES = {"Vocals": "vocals", "Karaoke": "karaoke", "De-Reverb": "dereverb", "De-Echo": "deecho",
                      "Denoise": "denoise"}


def test_every_model_has_valid_stages(catalog):
    from ai_cover_maker.core.pipeline.options import SEPARATION_STAGES

    data = json.loads((REPO_ROOT / "assets" / "models.json").read_text(encoding="utf-8"))
    missing = [m["id"] for m in data["models"] if "stages" not in m]
    assert not missing, f"Run tools/tag_model_stages.py --write, these models have no stages: {missing}"
    invalid = {m.id: m.stages for m in catalog if not set(m.stages) <= set(SEPARATION_STAGES)}
    assert not invalid
    assert all(catalog.for_stage(stage) for stage in SEPARATION_STAGES)
    assert not [m.id for m in catalog.by_arch("demucs") if m.stages]


def test_presets_and_recommended_models_fit_their_step(catalog):
    from ai_cover_maker.core.pipeline.options import SEPARATION_STAGES, load_presets

    wrong = [(name, stage, preset[stage]) for name, preset in load_presets().items() for stage in SEPARATION_STAGES
             if preset.get(stage) and stage not in catalog.find(preset[stage]).stages]
    assert not wrong, f"Preset models not tagged for their step: {wrong}"
    for stem, model_id in load_recommended(RECOMMENDED_FILE).items():
        if stem in RECOMMENDED_STAGES:
            assert RECOMMENDED_STAGES[stem] in catalog.find(model_id).stages, (stem, model_id)


def test_models_for_a_stage(tmp_path):
    (tmp_path / "catalog.json").write_text(json.dumps({"version": 2, "models": [
        {"id": "a", "arch": "roformer", "filename": "a.ckpt", "stages": ["dereverb", "deecho"]},
        {"id": "b", "arch": "roformer", "filename": "b.ckpt", "stages": []},
        {"id": "c", "arch": "roformer", "filename": "c.ckpt"}]}))
    catalog = ModelCatalog.from_file(tmp_path / "catalog.json")
    assert [m.id for m in catalog.for_stage("deecho")] == ["a"]
    assert catalog.find("a").stages == ("dereverb", "deecho") and catalog.find("c").stages == ()
    assert catalog.for_stage("vocals") == []
