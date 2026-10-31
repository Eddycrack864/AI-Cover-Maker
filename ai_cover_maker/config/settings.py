"""Separation parameters of each architecture.

The file is ``{section: {key: value}}`` where sections are the ``ArchSpec.settings_section``
names ("Roformer", "MDX23C", ...). The values are UVR5 UI's defaults and are not editable
from the app: every cover uses them.
"""

from pathlib import Path

from ai_cover_maker import paths
from ai_cover_maker.config.store import read_json


class SeparationSettings:
    def __init__(self, defaults_file: Path = None):
        self.defaults_file = Path(defaults_file or paths.DEFAULT_SETTINGS_FILE)

    def defaults(self) -> dict:
        return read_json(self.defaults_file)
