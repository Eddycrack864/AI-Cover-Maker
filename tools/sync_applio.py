"""Vendor the inference-only subset of Applio into third_party/applio.

Usage:
    python tools/sync_applio.py [--source .reference/Applio]

Copies INCLUDE from a local Applio checkout, applies PATCHES (exact text
replacements that must match the expected number of times, so an upstream
change fails loudly instead of silently) and records the upstream commit in
third_party/applio/UPSTREAM. Never edit third_party/applio by hand.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party" / "applio"

INCLUDE = [
    "LICENSE",
    "assets/config_template.json",
    "rvc/__init__.py",
    "rvc/configs/*.py",
    "rvc/configs/*.json",
    "rvc/infer/infer.py",
    "rvc/infer/pipeline.py",
    "rvc/lib/utils.py",
    "rvc/lib/user_config.py",
    "rvc/lib/zluda.py",
    "rvc/lib/algorithm/*.py",
    "rvc/lib/algorithm/generators/*.py",
    "rvc/lib/predictors/f0.py",
    "rvc/lib/predictors/RMVPE.py",
    "rvc/lib/predictors/Swift.py",
    "rvc/lib/predictors/chunking.py",
    "rvc/lib/tools/http_download.py",
    "rvc/lib/tools/device.py",
    "rvc/lib/tools/formant.py",
    "rvc/lib/tools/audio_cleanup.py",
    "rvc/lib/tools/gpu_checker.py",
    "rvc/lib/tools/model_download.py",
    "rvc/lib/tools/gdown.py",
]

EXCLUDE = {"rvc/lib/algorithm/discriminators.py"}

_CWD_SYS_PATH = "now_dir = os.getcwd()\nsys.path.append(now_dir)\n"
_PREDICTORS = 'os.path.join("rvc", "models", "predictors"'
_MODELS_ROOT = 'os.environ.get("APPLIO_MODELS_DIR", os.path.join("rvc", "models"))'

# (file, old, new, expected match count)
PATCHES = [
    # Do not append the caller's cwd to sys.path; the app adds third_party/applio.
    ("rvc/infer/infer.py", _CWD_SYS_PATH, "now_dir = os.getcwd()\n", 1),
    ("rvc/infer/pipeline.py", _CWD_SYS_PATH, "now_dir = os.getcwd()\n", 1),
    ("rvc/lib/utils.py", _CWD_SYS_PATH, "now_dir = os.getcwd()\n", 1),
    ("rvc/lib/tools/model_download.py", _CWD_SYS_PATH, "now_dir = os.getcwd()\n", 1),
    # Resolve predictor/embedder weights from APPLIO_MODELS_DIR instead of the cwd.
    ("rvc/lib/predictors/f0.py", _PREDICTORS, f'os.path.join({_MODELS_ROOT}, "predictors"', 3),
    ("rvc/lib/predictors/Swift.py", _PREDICTORS, f'os.path.join({_MODELS_ROOT}, "predictors"', 2),
    (
        "rvc/lib/utils.py",
        'embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")',
        'embedder_root = os.path.join(\n        os.environ.get("APPLIO_MODELS_DIR", os.path.join(now_dir, "rvc", "models")),\n        "embedders",\n    )',
        1,
    ),
]


def collect(source: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in INCLUDE:
        matches = sorted(source.glob(pattern))
        if not matches:
            raise SystemExit(f"Pattern matched nothing upstream: {pattern}")
        files += [m for m in matches if m.relative_to(source).as_posix() not in EXCLUDE]
    return files


def apply_patches() -> None:
    for rel, old, new, expected in PATCHES:
        path = DEST / rel
        text = path.read_text(encoding="utf-8")
        found = text.count(old)
        if found != expected:
            raise SystemExit(f"Patch for {rel} expected {expected} match(es), found {found}: {old!r}")
        path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def upstream_info(source: Path) -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        date = subprocess.check_output(["git", "-C", str(source), "log", "-1", "--format=%cs"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha, date = "unknown", "unknown"
    return f"repo: https://github.com/IAHispano/Applio\ncommit: {sha}\ndate: {date}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / ".reference" / "Applio")
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "rvc" / "infer" / "infer.py").is_file():
        raise SystemExit(f"Not an Applio checkout: {source}")

    files = collect(source)
    if DEST.exists():
        shutil.rmtree(DEST)
    for src in files:
        dst = DEST / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Normalize line endings so patches match regardless of git autocrlf.
        if src.suffix in {".py", ".json"}:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        else:
            shutil.copy2(src, dst)
    apply_patches()
    (DEST / "UPSTREAM").write_text(upstream_info(source), encoding="utf-8")
    print(f"Vendored {len(files)} files into {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
