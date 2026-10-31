"""Everything a cover run depends on, and the bundled presets."""

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from ai_cover_maker import paths
from ai_cover_maker.config.store import read_json
from ai_cover_maker.core.effects.chain import VocalEffects
from ai_cover_maker.core.rvc.engine import RvcParams

PRESETS_FILE_NAME = "pipeline_presets.json"
OUTPUT_FORMATS = ["wav", "flac", "mp3", "ogg", "m4a"]
# Separation stages in pipeline order. Each takes a catalog model id, or None to skip it.
SEPARATION_STAGES = ("vocals", "karaoke", "dereverb", "deecho", "denoise")


@dataclass(frozen=True)
class CoverOptions:
    voice: str  # name from core.rvc.voices.list_voices()
    vocals_model: str  # separates vocals from the instrumental (required)
    karaoke_model: str | None = None  # splits lead and backing vocals
    dereverb_model: str | None = None
    deecho_model: str | None = None
    denoise_model: str | None = None
    rvc: RvcParams = field(default_factory=RvcParams)
    backing_voice: str | None = None  # also convert the backing vocals (needs karaoke_model)
    backing_rvc: RvcParams = field(default_factory=RvcParams)
    effects: VocalEffects = field(default_factory=VocalEffects)
    # Transposes the whole song: added to the RVC pitch and applied to the instrumental and
    # to unconverted backing vocals. RVC's own pitch only moves the converted voice.
    key_shift: int = 0
    lead_gain_db: float = 0.0
    backing_gain_db: float = 0.0
    instrumental_gain_db: float = 0.0
    output_format: str = "wav"  # of the cover and of every stem saved next to it

    def stage_model(self, stage: str) -> str | None:
        return getattr(self, f"{stage}_model")


def _build(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def preset_options(preset: dict) -> dict:
    """``CoverOptions`` keyword arguments from a preset (no voice: that's the user's choice)."""
    preset = dict(preset)
    kwargs = {f"{stage}_model": preset.get(stage) for stage in SEPARATION_STAGES}
    if "rvc" in preset:
        kwargs["rvc"] = _build(RvcParams, preset["rvc"])
    if "effects" in preset:
        kwargs["effects"] = _build(VocalEffects, preset["effects"])
    return kwargs


def load_presets(path: Path = None) -> dict[str, dict]:
    """``{name: preset}`` from assets/pipeline_presets.json, in file order."""
    data = read_json(Path(path or paths.ASSETS_DIR / PRESETS_FILE_NAME))
    return dict(data["presets"])


def with_voice(options: CoverOptions, voice: str) -> CoverOptions:
    return replace(options, voice=voice)
