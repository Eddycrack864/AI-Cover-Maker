import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", help="also run the tests that need internet access")
    parser.addoption("--slow", action="store_true", help="also run the tests that load real models (minutes on CPU)")


def pytest_collection_modifyitems(config, items):
    for marker, option, reason in (
        ("network", "--network", "needs internet access: run pytest with --network"),
        ("slow", "--slow", "loads real models: run pytest with --slow"),
    ):
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def english_ui():
    """UI strings in English, whatever language is selected in the developer's own user_data."""
    from ai_cover_maker.i18n import I18nAuto, i18n

    i18n.reset(I18nAuto(language="en_US"))
    yield
    i18n.reset()


@pytest.fixture
def tmp_root(tmp_path):
    """A throwaway app root with a copy of the bundled assets (no user data)."""
    shutil.copytree(
        REPO_ROOT / "assets",
        tmp_path / "assets",
        ignore=shutil.ignore_patterns("*.py", "__pycache__", "config.json", "custom_settings.json"),
    )
    for name in ("models", "outputs", "ytdl"):
        (tmp_path / name).mkdir()
    return tmp_path
