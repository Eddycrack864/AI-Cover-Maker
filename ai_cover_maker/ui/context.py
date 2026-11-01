"""Shared dependencies handed to every tab renderer."""

from dataclasses import dataclass

from ai_cover_maker.config.settings import CoverSettings
from ai_cover_maker.config.store import ConfigStore, get_config
from ai_cover_maker.core.models.catalog import ModelCatalog, get_catalog


@dataclass
class AppContext:
    config: ConfigStore
    catalog: ModelCatalog
    cover_settings: CoverSettings

    @classmethod
    def create(cls) -> "AppContext":
        return cls(config=get_config(), catalog=get_catalog(), cover_settings=CoverSettings())
