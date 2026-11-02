"""Tag each model of assets/models.json with the cover steps it can be used for (``stages``).

The Cover tab lists, for each separation step, only the models tagged for it. A model can be
tagged for several steps (a de-reverb/de-echo model) or for none (drums, bass, crowd...).

A model's step is decided from, in this order (after ``OVERRIDES`` and ``EXCLUDED_ARCHS``):
1. Its name: "karaoke"/"BVE", "reverb", "echo", "denoise", or a stem the app never keeps (drums...).
2. The target stem of its config (Roformer/MDX23C ``training.target_instrument``).
3. "Inst"/"Instrumental" in its name: instrumental models are left out of the vocals step.
4. "Vocal"/"Voc"/"Main" in its name, or a "vocals" stem among the ones it outputs.
``OVERRIDES`` fixes the models these rules get wrong. ``BS-Roformer-Viperx-1053`` has no step: its
config targets "No Drum-Bass". A tagged model whose output stems are known
must have the stem the step keeps (``KEEP_LABELS`` in roles.py), or it is reported for review.

    python tools/tag_model_stages.py            # dry run: the proposed tags and what to review
    python tools/tag_model_stages.py --write    # store them in assets/models.json

Needs the app's environment and network: it downloads the models' small config files (cached).
"""

import argparse
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_cover_maker.core.errors import SeparationError  # noqa: E402
from ai_cover_maker.core.pipeline.options import SEPARATION_STAGES  # noqa: E402
from ai_cover_maker.core.pipeline.roles import normalize_label, split_stems  # noqa: E402
from tools.check_model_links import CATALOG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIGS = ROOT / "models" / "separation"  # configs of the models already downloaded
CACHE = Path(tempfile.gettempdir()) / "ai-cover-maker-model-configs"
CONFIG_ARCHS = ("roformer", "mdx23c")

# Name keywords, matched on the lowercased id and filename
STAGE_WORDS = {
    "karaoke": r"karaoke|kara|bve",
    "dereverb": r"reverb|deverb",
    "deecho": r"echo",
    "denoise": r"denoise",
}
# Stems no step keeps: these models are left out of every step
OTHER_WORDS = r"crowd|drum|bass|_other|female|aspiration|bleed.?suppressor|wind|mgm_"
INSTRUMENTAL_WORDS = r"inst(?!voc|vox)|\d+_(hp2?|sp)-uvr"  # VR HP/SP models keep the instrumental
VOCAL_WORDS = r"vocal|voc_|instvoc|instvox|main"

# Architectures left out of every step, with the reason
EXCLUDED_ARCHS = {"demucs": "its vocals are much worse than the other vocal models'"}

# id -> stages, for the models the rules get wrong (with the reason)
OVERRIDES = {
    # UVR's MDX-Net vocal models, named after their SDR score
    "UVR_MDXNET_1_9703.onnx": ["vocals"],
    "UVR_MDXNET_2_9682.onnx": ["vocals"],
    "UVR_MDXNET_3_9662.onnx": ["vocals"],
    "UVR_MDXNET_9482.onnx": ["vocals"],
    # Cleans instrumentals (its stems are Instrumental/Vocals), not vocals
    "MelBand Roformer | Denoise-Debleed by Gabox": [],
}


@dataclass
class Verdict:
    stages: list[str]
    reason: str
    stems: list[str] = field(default_factory=list)  # output stems, when known
    review: list[str] = field(default_factory=list)  # why a person should look at it


def classify(model: dict, config: dict | None, scored_stems: list[str] | None) -> Verdict:
    """The steps ``model`` can be used for, from its name, its config and its scored stems."""
    text = f"{model['id']} {model['filename']}".lower()
    training = (config or {}).get("training") or {}
    stems = [str(s) for s in (training.get("instruments") or scored_stems or [])]
    target = normalize_label(training.get("target_instrument"))
    normalized = {normalize_label(s) for s in stems}

    if model["id"] in OVERRIDES:
        verdict = Verdict(list(OVERRIDES[model["id"]]), "override")
    elif model["arch"] in EXCLUDED_ARCHS:
        verdict = Verdict([], f"{model['arch']}: {EXCLUDED_ARCHS[model['arch']]}")
    elif stages := [stage for stage, words in STAGE_WORDS.items() if re.search(words, text)]:
        verdict = Verdict(stages, "name")
    elif re.search(OTHER_WORDS, text):
        verdict = Verdict([], "name: not a stem the cover keeps")
    elif target in ("vocals", "vocal"):
        verdict = Verdict(["vocals"], "config target")
    elif target:
        verdict = Verdict([], f"config target: {training.get('target_instrument')}")
    elif re.search(INSTRUMENTAL_WORDS, text):
        verdict = Verdict([], "name: instrumental")
    elif re.search(VOCAL_WORDS, text):
        verdict = Verdict(["vocals"], "name")
    elif normalized & {"vocals", "vocal"}:
        verdict = Verdict(["vocals"], "stems")
    else:
        verdict = Verdict([], "unknown", review=["no rule matched"])
    verdict.stems = stems

    for stage in verdict.stages:
        if stems:
            try:
                split_stems(stage, {s: s for s in stems})
            except SeparationError:
                verdict.review.append(f"{stage}: no stem to keep among {stems}")
    if target in ("vocals", "vocal") and "vocals" not in verdict.stages and not verdict.stages:
        verdict.review.append("the config targets vocals but it is left out")
    return verdict


def config_url(model: dict) -> str | None:
    """The URL of the model's config file (the source that is not the model itself)."""
    for urls in model.get("sources", {}).values():
        for url in urls:
            if url.rsplit("/", 1)[-1] != model["filename"] and url.endswith((".yaml", ".yml")):
                return url
    return None


class _Loader(yaml.SafeLoader):
    """Safe loading that turns the python tags some configs use (e.g. tuples) into plain values."""


def _plain(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)


def load_config(url: str, cache: Path) -> dict | None:
    name = url.rsplit("/", 1)[-1]
    for path in (LOCAL_CONFIGS / name, cache / name):
        if path.exists():
            break
    else:
        cache.mkdir(parents=True, exist_ok=True)
        with urlopen(Request(url, headers={"User-Agent": "AI-Cover-Maker"}), timeout=30) as response:
            path.write_bytes(response.read())
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


def scored_stems() -> dict[str, list[str]]:
    """``{filename: stems}`` from the scores audio-separator ships."""
    with resources.files("audio_separator").joinpath("models-scores.json").open("r", encoding="utf-8") as f:
        return {name: data.get("stems") or [] for name, data in json.load(f).items()}


def tag_all(models: list[dict], cache: Path = CACHE) -> dict[str, Verdict]:
    urls = {m["id"]: config_url(m) for m in models if m["arch"] in CONFIG_ARCHS}
    unique = sorted({u for u in urls.values() if u})
    with ThreadPoolExecutor(8) as pool:
        configs = dict(zip(unique, pool.map(lambda u: _try_load(u, cache), unique)))
    scores = scored_stems()
    verdicts = {}
    for m in models:
        url = urls.get(m["id"])
        config = configs.get(url) if url else None
        verdict = classify(m, config if isinstance(config, dict) else None, scores.get(m["filename"]))
        if isinstance(config, Exception):
            verdict.review.append(f"config not loaded: {config}")
        verdicts[m["id"]] = verdict
    return verdicts


def _try_load(url: str, cache: Path):
    try:
        return load_config(url, cache)
    except Exception as e:  # reported per model
        return e


def report(models: list[dict], verdicts: dict[str, Verdict]) -> str:
    lines = []
    for stage in SEPARATION_STAGES:
        tagged = [m for m in models if stage in verdicts[m["id"]].stages]
        lines.append(f"\n== {stage} ({len(tagged)})")
        lines += [f"  {m['id']}  [{verdicts[m['id']].reason}]" for m in tagged]
    untagged = [m for m in models if not verdicts[m["id"]].stages]
    lines.append(f"\n== no step ({len(untagged)})")
    lines += [f"  {m['id']}  [{verdicts[m['id']].reason}]" for m in untagged]
    review = [m for m in models if verdicts[m["id"]].review]
    lines.append(f"\n== to review ({len(review)})")
    lines += [f"  {m['id']}: {'; '.join(verdicts[m['id']].review)}" for m in review]
    return "\n".join(lines)


def with_stages(model: dict, stages: list[str]) -> dict:
    """``model`` with ``stages`` right after ``filename`` (sources stay last)."""
    entry = {k: v for k, v in model.items() if k not in ("stages", "sources")}
    entry["stages"] = stages
    if "sources" in model:
        entry["sources"] = model["sources"]
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="store the tags in the catalog")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--cache", type=Path, default=CACHE, help="where the downloaded configs are kept")
    args = parser.parse_args(argv)

    data = json.loads(args.catalog.read_text(encoding="utf-8"))
    verdicts = tag_all(data["models"], args.cache)
    print(report(data["models"], verdicts))
    if args.write:
        data["models"] = [with_stages(m, verdicts[m["id"]].stages) for m in data["models"]]
        args.catalog.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                newline="\n")
        print(f"\nWrote the stages of {len(data['models'])} models to {args.catalog}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
