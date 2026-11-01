"""Saved settings: the fixed separation parameters, and the user's Cover tab settings."""

import logging
from pathlib import Path
from typing import Any

from ai_cover_maker import paths
from ai_cover_maker.config.store import read_json, write_json_atomic

logger = logging.getLogger(__name__)


class SeparationSettings:
    """Separation parameters of each architecture.

    The file is ``{section: {key: value}}`` where sections are the ``ArchSpec.settings_section``
    names ("Roformer", "MDX23C", ...). The values are UVR5 UI's defaults and are not editable
    from the app: every cover uses them.
    """

    def __init__(self, defaults_file: Path = None):
        self.defaults_file = Path(defaults_file or paths.DEFAULT_SETTINGS_FILE)

    def defaults(self) -> dict:
        return read_json(self.defaults_file)


class CoverSettings:
    """The Cover tab settings the user saved, ``{handler parameter name: value}``.

    They are loaded when the app starts; resetting deletes the file, so the defaults are used again.
    """

    def __init__(self, file: Path = None):
        self.file = Path(file or paths.COVER_SETTINGS_FILE)

    def load(self) -> dict[str, Any]:
        if not self.file.exists():
            return {}
        try:
            data = read_json(self.file)
        except (OSError, ValueError) as e:
            logger.warning("Ignoring the saved Cover settings in '%s': %s", self.file, e)
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, values: dict[str, Any]) -> None:
        write_json_atomic(self.file, values, indent=4)

    def reset(self) -> None:
        self.file.unlink(missing_ok=True)
