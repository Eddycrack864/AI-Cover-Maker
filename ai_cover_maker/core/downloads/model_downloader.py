"""Separation model downloads, before a model is loaded.

audio-separator's own download is tried first. When it fails, the catalog URLs are tried:
GitHub, then Hugging Face. Those downloads are resumable: an interrupted one keeps its
``.part`` file and the next attempt continues it.
"""

import http.client
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from ai_cover_maker import paths
from ai_cover_maker.core.errors import DownloadError
from ai_cover_maker.core.models.catalog import SOURCES as CATALOG_SOURCES
from ai_cover_maker.core.models.catalog import Model

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

RETRIES = 5  # attempts per URL; each one resumes from the partial file
_RETRY_DELAY = 2  # seconds, multiplied by the attempt number
_TIMEOUT = 60
_CHUNK = 1024 * 1024
# Network errors worth retrying (HTTPError is handled apart: most 4xx are final)
_RETRY_ERRORS = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


def _noop_progress(fraction: float, desc: str) -> None:
    pass


def filename_from_url(url: str) -> str:
    return os.path.basename(urllib.parse.urlparse(url).path)


def part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def _fetch(url: str, part: Path, on_fraction: Callable[[float], None]) -> None:
    """Download ``url`` into ``part``, continuing from its current size when the server allows it."""
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        if done and getattr(response, "status", None) != 206:
            done = 0  # the server ignored the range: start over
        length = int(response.headers.get("Content-Length") or 0)
        total = done + length
        with open(part, "ab" if done else "wb") as f:
            while chunk := response.read(_CHUNK):
                f.write(chunk)
                done += len(chunk)
                if total:
                    on_fraction(done / total)
    if length and done < total:
        raise ConnectionError(f"connection closed after {done} of {total} bytes")


def _download_python(url: str, dest: Path, on_fraction: Callable[[float], None]) -> None:
    part = part_path(dest)
    for attempt in range(1, RETRIES + 1):
        try:
            _fetch(url, part, on_fraction)
            os.replace(part, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # the partial file doesn't match this one: start over
                part.unlink(missing_ok=True)
            elif e.code < 500:
                raise
            error = e
        except _RETRY_ERRORS as e:
            error = e
        if attempt < RETRIES:
            logger.warning("Download of %s interrupted (%s), retrying (%d/%d)", url, error, attempt, RETRIES)
            time.sleep(_RETRY_DELAY * attempt)
    # The partial file is kept so the next attempt resumes it
    raise error


def file_urls(model: Model, source: str) -> dict[str, list[str]]:
    """Each file of ``model`` with the URLs to try: ``source`` first, then the other sources."""
    files = {filename_from_url(url): [url] for url in model.urls(source)}
    for other in CATALOG_SOURCES:
        for url in model.sources.get(other) or []:
            urls = files.get(filename_from_url(url))
            if urls is not None and url not in urls:
                urls.append(url)
    return files


def download_model(
    model: Model,
    source: str = "github",
    *,
    models_dir: Path = None,
    progress: ProgressFn = _noop_progress,
) -> Iterator[str]:
    """Download every file of ``model`` (weights + config), ``source`` first. Yields one log line at a time."""
    files = file_urls(model, source)
    if not files:
        yield f"No URLs found for model '{model.id}' using source '{source}'."
        return

    models_dir = Path(models_dir or paths.SEPARATION_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    progress(0, "Starting downloads...")

    for i, (filename, urls) in enumerate(files.items()):
        dest = models_dir / filename
        if dest.exists():
            yield f"{filename} already exists."
            continue

        def on_fraction(fraction, i=i, filename=filename):
            progress((i + fraction) / total, f"File {i + 1}/{total}: {filename} ({int(fraction * 100)}%)")

        progress(i / total, f"Starting download of {filename} ({i + 1}/{total})")
        for attempt, url in enumerate(urls):
            host = urllib.parse.urlparse(url).netloc
            if attempt:
                # A partial file from another source can't be resumed safely
                part_path(dest).unlink(missing_ok=True)
                yield f"Trying another source for {filename}..."
            yield f"Downloading {filename} from {host}..."
            try:
                _download_python(url, dest, on_fraction)
                yield f"{filename} downloaded successfully!"
                break
            except Exception as e:
                logger.warning("Download of %s failed: %s", url, e)
                yield f"Error downloading {filename} from {host}: {e}"

    progress(1.0, "Download process completed")


def ensure_downloaded(model: Model, models_dir: Path = None, progress: ProgressFn = None,
                      native: Callable[[], object] = None) -> None:
    """Download the missing files of a catalog model before audio-separator loads it.

    ``native`` is audio-separator's own download (see ``runtime.atomic_downloads``). It is skipped
    when one of our downloads was interrupted, so that one resumes instead of starting over.
    """
    models_dir = Path(models_dir or paths.SEPARATION_MODELS_DIR)
    progress = progress or _noop_progress

    def missing() -> list[str]:
        return [name for name in file_urls(model, "github") if not (models_dir / name).exists()]

    if not missing():
        return
    if native is not None and not any(part_path(models_dir / name).exists() for name in missing()):
        progress(0.0, f"Downloading {model.id} with audio-separator...")
        try:
            native()
        except Exception as e:
            logger.warning("audio-separator could not download %s (%s), trying GitHub and Hugging Face",
                           model.filename, e)
        if not missing():
            return
    for line in download_model(model, "github", models_dir=models_dir, progress=progress):
        logger.info(line)
    if failed := missing():
        raise DownloadError(f"could not download {', '.join(failed)}. Run it again to resume the download")
