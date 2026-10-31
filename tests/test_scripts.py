"""Guards for the distribution scripts."""

import hashlib

from tests.conftest import REPO_ROOT

# cmd.exe keeps reading a running .bat at the same byte offset, and the updater rewrites itself with
# "git reset". Older updaters resume right after the reset block, so the part above the marker must
# stay byte-identical forever (.gitattributes keeps .bat files CRLF on every platform).
UPDATER_MARKER = b"rem ===== Do not edit anything above this line. ====="
UPDATER_PREFIX_SHA256 = "4a655e47efc30e3d0db1c787d1c59618fe204d19013921bfb0f64ecd51d4d7f8"


def test_windows_updater_prefix_is_unchanged():
    data = (REPO_ROOT / "AI-Cover-Maker-updater.bat").read_bytes()
    prefix, marker, _ = data.partition(UPDATER_MARKER)
    assert marker, "the 'Do not edit' marker is missing from AI-Cover-Maker-updater.bat"
    assert hashlib.sha256(prefix).hexdigest() == UPDATER_PREFIX_SHA256


def test_installers_share_versions():
    bat = (REPO_ROOT / "AI-Cover-Maker-installer.bat").read_text(encoding="utf-8")
    sh = (REPO_ROOT / "AI-Cover-Maker-installer.sh").read_text(encoding="utf-8")
    for name in ("PYTHON_VERSION", "PIP_VERSION", "TORCH_VERSION", "TORCHAUDIO_VERSION", "TORCH_INDEX"):
        bat_value = next(line for line in bat.splitlines() if line.startswith(f'set "{name}='))
        sh_value = next(line for line in sh.splitlines() if line.startswith(f"{name}="))
        assert bat_value.split("=", 1)[1].rstrip('"') == sh_value.split("=", 1)[1].strip('"'), name


def test_torch_constraints_match_the_installers():
    sh = (REPO_ROOT / "AI-Cover-Maker-installer.sh").read_text(encoding="utf-8")
    version = {name: next(line for line in sh.splitlines() if line.startswith(f"{name}=")).split("=", 1)[1].strip('"')
               for name in ("TORCH_VERSION", "TORCHAUDIO_VERSION")}
    constraints = (REPO_ROOT / "constraints.txt").read_text(encoding="utf-8")
    assert f"torch=={version['TORCH_VERSION']}" in constraints.splitlines()
    assert f"torchaudio=={version['TORCHAUDIO_VERSION']}" in constraints.splitlines()
