"""Pretrained files RVC inference needs: pitch predictors and content embedders.

Applio downloads embedders on first use but not RMVPE/FCPE, and its downloads can't
resume. Everything is fetched here up front with the resumable downloader instead.
"""

import logging
from pathlib import Path

from ai_cover_maker import paths
from ai_cover_maker.core.downloads.model_downloader import _download_python, part_path
from ai_cover_maker.core.errors import DownloadError

logger = logging.getLogger(__name__)

HF_RESOURCES = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

# f0 method -> predictor weights it needs (crepe ships inside torchcrepe)
PREDICTORS = {"rmvpe": ["rmvpe.pt"], "fcpe": ["fcpe.pt"], "swift": ["swift.onnx"], "crepe": []}

# embedder choice -> folder name on the Hub and in RVC_ASSETS_DIR/embedders
EMBEDDERS = {
    "contentvec": "contentvec",
    "spin": "spin",
    "spin-v2": "spin-v2",
    "chinese-hubert-base": "chinese_hubert_base",
    "japanese-hubert-base": "japanese_hubert_base",
    "korean-hubert-base": "korean_hubert_base",
}
EMBEDDER_FILES = ("pytorch_model.bin", "config.json")


def required_files(f0_method: str, embedder: str) -> dict[str, Path]:
    """``{url: destination}`` of every file a conversion with these settings needs."""
    root = paths.RVC_ASSETS_DIR
    files = {f"{HF_RESOURCES}/predictors/{name}": root / "predictors" / name for name in PREDICTORS.get(f0_method, [])}
    if embedder in EMBEDDERS:  # "custom" embedders are user files
        folder = EMBEDDERS[embedder]
        for name in EMBEDDER_FILES:
            files[f"{HF_RESOURCES}/embedders/{folder}/{name}"] = root / "embedders" / folder / name
    return files


def missing_files(f0_method: str, embedder: str) -> dict[str, Path]:
    return {url: dest for url, dest in required_files(f0_method, embedder).items() if not dest.is_file()}


def ensure_assets(f0_method: str, embedder: str, progress=None) -> None:
    """Download the predictor and embedder weights that are not there yet."""
    missing = missing_files(f0_method, embedder)
    for i, (url, dest) in enumerate(missing.items()):
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", url)

        def on_fraction(fraction, i=i, name=dest.name):
            if progress:
                progress((i + fraction) / len(missing), f"Downloading {name} ({int(fraction * 100)}%)")

        try:
            _download_python(url, dest, on_fraction)
        except Exception as e:
            raise DownloadError(
                f"could not download {dest.name} ({e}). Run it again to resume the download"
            ) from e
        finally:
            if dest.is_file():
                part_path(dest).unlink(missing_ok=True)
