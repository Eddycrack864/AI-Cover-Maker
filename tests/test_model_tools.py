"""Offline tests of tools/check_model_links.py and tools/sync_models.py (no network: links are faked)."""

from tools.check_model_links import (
    GITHUB_MODEL_CONFIGS, HF_MIRROR, LinkResult, build_report, huggingface_urls, model_files, print_report, summarize,
)
from tools.sync_models import arch_for, id_for, plan, working_sources
from tools.tag_model_stages import Verdict

GH = GITHUB_MODEL_CONFIGS


def fake_checker(working: set[str]):
    return lambda url: LinkResult(url, url in working, 200 if url in working else 404)


def roformer(filename, sources=None, model_id=None):
    return {"id": model_id or filename, "arch": "roformer", "filename": filename, "sources": sources or {}}


def test_files_and_mirror_paths():
    model = roformer("a b.ckpt", {"github": [f"{GH}/a%20b.ckpt", f"{GH}/a.yaml"]})
    assert model_files(model) == ["a%20b.ckpt", "a.yaml"]
    assert huggingface_urls({"arch": "mdxnet"}, ["x.onnx"]) == [f"{HF_MIRROR}/mdx-net/x.onnx"]
    assert huggingface_urls({"arch": "vrarch"}, ["1 x.pth"]) == [f"{HF_MIRROR}/vr-arch/1%20x.pth"]
    assert huggingface_urls({"arch": "demucs"}, ["x.yaml"]) == []
    assert huggingface_urls(roformer("none.ckpt")) == []


def test_report(capsys):
    ok = roformer("ok.ckpt", {"github": [f"{GH}/ok.ckpt"], "huggingface": [f"{HF_MIRROR}/roformers/ok.ckpt"]})
    broken = roformer("broken.ckpt", {"github": [f"{GH}/broken.ckpt"], "huggingface": [f"{HF_MIRROR}/roformers/broken.ckpt"]})
    pending = roformer("pending.ckpt", {"github": [f"{GH}/pending.ckpt", f"{GH}/shared.yaml"]})
    uploaded = roformer("up.ckpt", {"github": [f"{GH}/up.ckpt"]})
    working = {f"{GH}/ok.ckpt", f"{HF_MIRROR}/roformers/ok.ckpt", f"{GH}/broken.ckpt", f"{GH}/pending.ckpt",
               f"{GH}/shared.yaml", f"{HF_MIRROR}/roformers/shared.yaml", f"{GH}/up.ckpt", f"{HF_MIRROR}/roformers/up.ckpt"}

    report = build_report([ok, broken, pending, uploaded], checker=fake_checker(working), workers=2)

    assert [summarize(e) for e in report] == [
        {"github": "ok", "huggingface": "ok"},
        {"github": "ok", "huggingface": "broken"},
        {"github": "ok", "huggingface": "none", "mirror": "not uploaded"},
        {"github": "ok", "huggingface": "none", "mirror": "uploaded"},
    ]
    assert print_report(report) == 1
    out = capsys.readouterr().out
    assert "ok.ckpt]" not in out  # working models are only listed with --verbose
    assert f"BROKEN 404" in out and "NOT on the mirror" in out and "run tools/sync_models.py --write" in out


def test_arch_and_id():
    assert arch_for("MDXC", "Roformer Model: BS Roformer | Karaoke by anvuew") == "roformer"
    assert arch_for("MDXC", "MDX23C Model: MDX23C-InstVoc HQ") == "mdx23c"
    assert (arch_for("MDX", "x"), arch_for("VR", "x"), arch_for("Demucs", "x")) == ("mdxnet", "vrarch", "demucs")
    assert id_for("roformer", "Roformer Model: BS Roformer | Karaoke by anvuew", "f.ckpt") == "BS Roformer | Karaoke by anvuew"
    assert id_for("mdxnet", "MDX-Net Model: UVR-MDX-NET Inst HQ 3", "UVR-MDX-NET-Inst_HQ_3.onnx") == "UVR-MDX-NET-Inst_HQ_3.onnx"


def test_working_sources():
    files = ["m.ckpt", "m.yaml"]
    uvr_yaml = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mdx_model_data/mdx_c_configs/m.yaml"
    checker = fake_checker({f"{GH}/m.ckpt", uvr_yaml, f"{HF_MIRROR}/roformers/m.ckpt"})
    # GitHub falls back to UVR's repo; Hugging Face is left out until every file is on the mirror
    assert working_sources("roformer", files, checker) == {"github": [f"{GH}/m.ckpt", uvr_yaml]}
    assert working_sources("roformer", files, fake_checker(set())) == {}
    assert working_sources("demucs", ["https://dl.fbaipublicfiles.com/x.th", "x.yaml"], checker) == {}


def test_plan_inserts_after_upstream_predecessor_and_fills_sources():
    catalog = [
        roformer("a.ckpt", {"github": [f"{GH}/a.ckpt"], "huggingface": [f"{HF_MIRROR}/roformers/a.ckpt"]}),
        roformer("c.ckpt", {"github": [f"{GH}/c.ckpt"]}),
        roformer("sw.ckpt"),
        {"id": "x.onnx", "arch": "mdxnet", "filename": "x.onnx", "sources": {}},
    ]
    upstream = [
        {"name": "Roformer Model: A", "type": "MDXC", "filename": "a.ckpt", "files": ["a.ckpt"]},
        {"name": "Roformer Model: New B", "type": "MDXC", "filename": "b.ckpt", "files": ["b.ckpt"]},
        {"name": "Roformer Model: C", "type": "MDXC", "filename": "c.ckpt", "files": ["c.ckpt"]},
        {"name": "Roformer Model: SW", "type": "MDXC", "filename": "sw.ckpt", "files": ["sw.ckpt"]},
        {"name": "MDX-Net Model: Y", "type": "MDX", "filename": "y.onnx", "files": ["y.onnx"]},
    ]
    working = {f"{GH}/b.ckpt", f"{HF_MIRROR}/roformers/b.ckpt", f"{HF_MIRROR}/roformers/c.ckpt", f"{GH}/sw.ckpt",
               f"{GH}/y.onnx"}

    tags = {"New B": Verdict(["vocals"], "name"), "y.onnx": Verdict([], "unknown", review=["no rule matched"])}
    models, changes = plan(catalog, upstream, fake_checker(working), tag=lambda entry: tags[entry["id"]])

    assert [m["filename"] for m in models] == ["a.ckpt", "b.ckpt", "c.ckpt", "sw.ckpt", "x.onnx", "y.onnx"]
    assert models[1] == {"id": "New B", "arch": "roformer", "filename": "b.ckpt", "stages": ["vocals"],
                         "sources": {"github": [f"{GH}/b.ckpt"], "huggingface": [f"{HF_MIRROR}/roformers/b.ckpt"]}}
    assert models[2]["sources"]["huggingface"] == [f"{HF_MIRROR}/roformers/c.ckpt"]
    assert models[3]["sources"] == {"github": [f"{GH}/sw.ckpt"]}
    assert models[5]["sources"] == {"github": [f"{GH}/y.onnx"]}
    assert changes == [
        "fill huggingface of [roformer] c.ckpt",
        "fill github of [roformer] sw.ckpt",
        "add [roformer] New B (steps: vocals)",
        "add [mdxnet] y.onnx (no huggingface source yet; steps: none; "
        "review the steps in tools/tag_model_stages.py: no rule matched)",
    ]
    assert catalog[1]["sources"] == {"github": [f"{GH}/c.ckpt"]}  # the input is not modified
    assert plan(models, upstream, fake_checker(working))[1] == []


def _model(model_id, arch="roformer", filename=None):
    return {"id": model_id, "arch": arch, "filename": filename or f"{model_id}.ckpt"}


def _config(target, instruments=("Vocals", "Instrumental")):
    return {"training": {"target_instrument": target, "instruments": list(instruments)}}


def test_stage_tags_come_from_the_name_the_config_and_the_stems():
    from tools.tag_model_stages import classify

    def stages(model, config=None, stems=None):
        return classify(model, config, stems).stages

    assert stages(_model("BS Roformer | Karaoke by x"), _config("Vocals")) == ["karaoke"]
    assert stages(_model("UVR-DeEcho-DeReverb.pth", "vrarch"), stems=["Dry", "No Dry"]) == ["dereverb", "deecho"]
    assert stages(_model("Reverb_HQ_By_FoxJoy.onnx", "mdxnet")) == ["dereverb"]
    # The config's target decides between vocal and instrumental models, whatever the name
    assert stages(_model("MelBand Roformer | Big Beta 6 by unwa"), _config("vocals")) == ["vocals"]
    assert stages(_model("MelBand Roformer | Vocals FV1 by x"), _config("Instrumental")) == []
    # Only vocal models in the vocals step; models trained on both stems are vocal models too
    assert stages(_model("UVR-MDX-NET-Inst_Main.onnx", "mdxnet")) == []
    assert stages(_model("MDX23C-8KFFT-InstVoc_HQ.ckpt", "mdx23c")) == ["vocals"]
    assert stages(_model("UVR-MDX-NET_Main_340.onnx", "mdxnet")) == ["vocals"]
    assert stages(_model("BS Roformer SW by x"), _config(None, ["bass", "drums", "vocals"])) == ["vocals"]
    assert stages(_model("htdemucs.yaml", "demucs"), stems=["vocals", "drums", "bass", "other"]) == []
    assert stages(_model("9_HP2-UVR.pth", "vrarch"), stems=["vocals", "instrumental"]) == []
    assert stages(_model("kuielab_a_drums.onnx", "mdxnet"), stems=["drums", "nodrums"]) == []
    # Overrides win over the rules
    assert stages(_model("UVR_MDXNET_9482.onnx", "mdxnet")) == ["vocals"]


def test_stage_tags_report_what_needs_a_look():
    from tools.tag_model_stages import classify

    # A denoise model whose stems the denoise step can't keep from
    verdict = classify(_model("Some Denoise model"), _config(None, ["Instrumental", "Vocals"]), None)
    assert verdict.stages == ["denoise"] and verdict.review
    assert classify(_model("mystery.onnx", "mdxnet"), None, None).review == ["no rule matched"]
    assert not classify(_model("UVR-DeNoise.pth", "vrarch"), None, ["Noise", "No Noise"]).review


def test_stages_go_after_the_filename():
    from tools.tag_model_stages import with_stages

    entry = with_stages({"id": "a", "arch": "x", "filename": "a.ckpt", "sources": {}, "stages": ["old"]}, ["vocals"])
    assert list(entry) == ["id", "arch", "filename", "stages", "sources"] and entry["stages"] == ["vocals"]
