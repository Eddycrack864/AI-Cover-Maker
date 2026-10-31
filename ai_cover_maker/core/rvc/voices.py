"""Voice models: ``models/rvc/<folder>/<model>.pth`` plus an optional ``.index``.

A folder may hold several checkpoints of the same voice (e.g. different epochs); each
one is listed as its own voice and they share the folder's index.
"""

import logging
import os
import re
import shutil
import tempfile
import threading
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ai_cover_maker import paths
from ai_cover_maker.core.errors import DownloadError

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = (".pth", ".onnx")
INDEX_EXTENSION = ".index"
IMPORT_EXTENSIONS = (*MODEL_EXTENSIONS, INDEX_EXTENSION, ".zip")
# Applio's downloader keeps its archives here
IGNORED_FOLDERS = {"zips"}

_download_lock = threading.Lock()  # Applio's downloader changes the process cwd


@dataclass(frozen=True)
class Voice:
    name: str  # what the user picks: "<folder>" or "<folder>/<checkpoint>"
    model_path: Path
    index_path: Path | None = None


def _is_model_file(path: Path) -> bool:
    # G_*/D_* are training checkpoints, not inference weights
    return path.suffix.lower() in MODEL_EXTENSIONS and not re.match(r"^[GD]_", path.name)


def _pick_index(folder: Path) -> Path | None:
    # "trained_*" indexes are intermediate files; Applio itself swaps them for "added_*"
    indexes = sorted(p for p in folder.glob(f"*{INDEX_EXTENSION}") if "trained" not in p.name.lower())
    if not indexes:
        return None
    added = [p for p in indexes if "added" in p.name.lower()]
    return (added or indexes)[0]


def list_voices(voices_dir: Path = None) -> list[Voice]:
    voices_dir = Path(voices_dir or paths.VOICE_MODELS_DIR)
    if not voices_dir.is_dir():
        return []
    voices = []
    for folder in sorted((p for p in voices_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        if folder.name in IGNORED_FOLDERS:
            continue
        models = sorted((p for p in folder.iterdir() if p.is_file() and _is_model_file(p)), key=lambda p: p.name.lower())
        index = _pick_index(folder)
        for model in models:
            name = folder.name if len(models) == 1 else f"{folder.name}/{model.stem}"
            voices.append(Voice(name, model, index))
    return voices


def get_voice(name: str, voices_dir: Path = None) -> Voice:
    for voice in list_voices(voices_dir):
        if voice.name == name:
            return voice
    raise KeyError(f"Voice model not found: {name}")


def safe_folder_name(name: str) -> str:
    """A folder name without path separators or characters that are invalid on Windows."""
    name = unicodedata.normalize("NFC", name or "").strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name[:100]


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        for member in zf.infolist():
            target = (root / member.filename.replace("\\", "/")).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise DownloadError(f"The archive {archive.name} contains an unsafe file path")
        zf.extractall(destination)


def _collect(staging: Path) -> list[Path]:
    """Extract every archive in ``staging`` and return the model and index files found."""
    for archive in list(staging.rglob("*.zip")):
        try:
            _safe_extract(archive, archive.with_suffix(""))
        except zipfile.BadZipFile as e:
            raise DownloadError(f"{archive.name} is not a valid zip file") from e
        archive.unlink()
    return [p for p in staging.rglob("*")
            if p.is_file() and "__MACOSX" not in p.parts and (_is_model_file(p) or p.suffix.lower() == INDEX_EXTENSION)]


def install_voice(files: list[Path], name: str, voices_dir: Path = None, overwrite: bool = False) -> Path:
    """Copy model/index files (or zips containing them) into ``models/rvc/<name>/``."""
    voices_dir = Path(voices_dir or paths.VOICE_MODELS_DIR)
    folder_name = safe_folder_name(name)
    if not folder_name:
        raise DownloadError("Please write a name for the voice model")
    files = [Path(f) for f in files if f]
    if not files:
        raise DownloadError("No files provided")
    if unsupported := [f.name for f in files if f.suffix.lower() not in IMPORT_EXTENSIONS]:
        raise DownloadError(f"Unsupported files: {', '.join(unsupported)} (use .pth, .onnx, .index or .zip)")

    with tempfile.TemporaryDirectory(prefix="voice-") as tmp:
        staging = Path(tmp)
        for file in files:
            shutil.copy2(file, staging / file.name)
        found = _collect(staging)
        if not any(_is_model_file(p) for p in found):
            raise DownloadError("No voice model (.pth or .onnx) found in the provided files")
        return _move_into(found, voices_dir / folder_name, overwrite)


def _move_into(found: list[Path], folder: Path, overwrite: bool) -> Path:
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise DownloadError(f"A voice model named '{folder.name}' already exists")
    folder.mkdir(parents=True, exist_ok=True)
    for file in found:
        shutil.move(str(file), str(folder / file.name))
    logger.info("Installed voice model into %s", folder)
    return folder


def download_voice(url: str, name: str = "", voices_dir: Path = None) -> Path:
    """Download a voice from a link (Hugging Face, Google Drive or a direct link to a .zip/.pth/.index).

    The download itself is Applio's (it knows how to fetch from those hosts); the files are
    then installed like an upload. ``name`` defaults to the downloaded file's name.
    """
    url = (url or "").strip()
    if not url:
        raise DownloadError("No link provided")
    voices_dir = Path(voices_dir or paths.VOICE_MODELS_DIR)
    voices_dir.mkdir(parents=True, exist_ok=True)

    from ai_cover_maker.core.rvc.engine import import_applio

    downloader = import_applio("rvc.lib.tools.model_download")
    with _download_lock, tempfile.TemporaryDirectory(prefix="voice-dl-") as tmp:
        cwd = os.getcwd()
        previous, downloader.zips_path = downloader.zips_path, tmp
        try:
            result = downloader.download_from_url(url)
        finally:
            downloader.zips_path = previous
            os.chdir(cwd)
        downloaded = [p for p in Path(tmp).iterdir() if p.is_file()]
        if result != "downloaded" or not downloaded:
            raise DownloadError(f"Could not download the voice model from {url}, check the console for details")
        found = _collect(Path(tmp))
        if not any(_is_model_file(p) for p in found):
            raise DownloadError("No voice model (.pth or .onnx) found in the download")
        default_name = next((p.stem for p in downloaded if p.suffix.lower() == ".zip"), None) \
            or next(p.stem for p in found if _is_model_file(p))
        return _move_into(found, voices_dir / (safe_folder_name(name) or safe_folder_name(default_name)), False)


def delete_voice(name: str, voices_dir: Path = None) -> None:
    """Delete a voice folder (``name`` is the folder, or a voice name inside it)."""
    voices_dir = Path(voices_dir or paths.VOICE_MODELS_DIR)
    folder = (voices_dir / name.split("/", 1)[0]).resolve()
    if folder.parent != voices_dir.resolve() or not folder.is_dir() or folder.name in IGNORED_FOLDERS:
        raise DownloadError(f"Voice model not found: {name}")
    shutil.rmtree(folder)
