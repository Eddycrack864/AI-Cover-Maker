"""Filesystem locations, resolved from the package location instead of the CWD."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("AI_COVER_MAKER_ROOT") or Path(__file__).resolve().parents[1])

ASSETS_DIR = ROOT / "assets"
LANG_DIR = ASSETS_DIR / "i18n" / "languages"
THEMES_DIR = ASSETS_DIR / "themes"
FAVICON = ASSETS_DIR / "favicon.ico"
DEFAULT_CONFIG_FILE = ASSETS_DIR / "config.default.json"
DEFAULT_SETTINGS_FILE = ASSETS_DIR / "default_settings.json"
MODELS_CATALOG_FILE = ASSETS_DIR / "models.json"

USER_DATA_DIR = ROOT / "user_data"
USER_CONFIG_FILE = USER_DATA_DIR / "config.json"
CUSTOM_SETTINGS_FILE = USER_DATA_DIR / "custom_settings.json"
COVER_SETTINGS_FILE = USER_DATA_DIR / "cover_settings.json"
APPLIO_CONFIG_FILE = USER_DATA_DIR / "applio_config.json"

# Pre-user_data locations, migrated into USER_DATA_DIR on startup
LEGACY_CONFIG_FILE = ASSETS_DIR / "config.json"
LEGACY_CUSTOM_SETTINGS_FILE = ASSETS_DIR / "custom_settings.json"

# Vendored Applio inference code (see tools/sync_applio.py)
APPLIO_DIR = ROOT / "third_party" / "applio"

MODELS_DIR = ROOT / "models"
SEPARATION_MODELS_DIR = MODELS_DIR / "separation"
VOICE_MODELS_DIR = MODELS_DIR / "rvc"  # one folder per voice: <name>/<name>.pth (+ .index)
RVC_ASSETS_DIR = MODELS_DIR / "rvc_assets"  # predictors/ (rmvpe, fcpe) and embedders/
WORKSPACE_DIR = ROOT / "workspace"  # cached pipeline stages, one folder per input song
OUTPUTS_DIR = ROOT / "outputs"
YTDL_DIR = ROOT / "ytdl"
