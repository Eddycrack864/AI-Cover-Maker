"""Content-addressed cache of pipeline stage outputs.

Layout: ``workspace/<song>-<content hash>/<stage>-<key>/`` with the stage's audio files
and a ``manifest.json`` mapping each role (e.g. ``vocals``) to its file. A stage key
hashes everything its output depends on (model, parameters, the keys of its inputs),
so changing any setting recomputes exactly the stages it affects and nothing else.
"""

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from ai_cover_maker import paths
from ai_cover_maker.config.store import read_json, write_json_atomic

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
_CHUNK = 1024 * 1024


def file_hash(path: Path) -> str:
    """SHA-256 of a file's content (not its name or date)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path | None) -> str | None:
    """Cheap identity of a large file (voice models): name, size and modification time."""
    if not path:
        return None
    stat = Path(path).stat()
    return f"{Path(path).name}:{stat.st_size}:{stat.st_mtime_ns}"


def stage_key(stage: str, payload: Any) -> str:
    data = json.dumps({"stage": stage, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", name).strip("._")[:60] or "song"


class SongWorkspace:
    """Cached stages of one input song."""

    def __init__(self, song: Path, root: Path = None):
        self.song = Path(song)
        self.song_key = file_hash(self.song)[:16]
        self.dir = Path(root or paths.WORKSPACE_DIR) / f"{_safe_name(self.song.stem)}-{self.song_key}"

    def stage_dir(self, stage: str, key: str) -> Path:
        return self.dir / f"{stage}-{key}"

    def load(self, stage: str, key: str) -> dict[str, Path] | None:
        """The cached ``{role: file}`` of a stage, or None if it was never completed."""
        folder = self.stage_dir(stage, key)
        try:
            manifest = read_json(folder / MANIFEST)
        except (OSError, ValueError):
            return None
        files = {role: folder / name for role, name in manifest.get("files", {}).items()}
        if not files or not all(f.is_file() for f in files.values()):
            return None
        return files

    def store(self, stage: str, key: str, files: dict[str, Path], move: bool = True) -> dict[str, Path]:
        """Put the stage's output files in its folder and mark the stage as complete."""
        folder = self.stage_dir(stage, key)
        if folder.exists():
            shutil.rmtree(folder)  # leftovers of an interrupted run
        folder.mkdir(parents=True)
        stored = {}
        for role, src in files.items():
            dest = folder / f"{role}{Path(src).suffix.lower()}"
            if move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(src, dest)
            stored[role] = dest
        # The manifest is written last: a stage folder without it is incomplete
        write_json_atomic(folder / MANIFEST, {"stage": stage, "key": key, "created": time.time(),
                                             "files": {role: p.name for role, p in stored.items()}})
        return stored


def clear_workspace(root: Path = None) -> int:
    """Delete every cached stage. Returns the number of song folders removed."""
    root = Path(root or paths.WORKSPACE_DIR)
    if not root.is_dir():
        return 0
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


def workspace_size(root: Path = None) -> int:
    root = Path(root or paths.WORKSPACE_DIR)
    if not root.is_dir():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

