"""Shared dependencies handed to every tab renderer."""

from dataclasses import dataclass, field

from ai_cover_maker.config.settings import SeparationSettings
from ai_cover_maker.config.store import ConfigStore, get_config
from ai_cover_maker.core.models.catalog import ModelCatalog, get_catalog


@dataclass
class AppContext:
    config: ConfigStore
    settings: SeparationSettings
    catalog: ModelCatalog
    initial_settings: dict = field(default_factory=dict)

    @classmethod
    def create(cls) -> "AppContext":
        config = get_config()
        settings = SeparationSettings(config)
        return cls(config=config, settings=settings, catalog=get_catalog(), initial_settings=settings.initial())
