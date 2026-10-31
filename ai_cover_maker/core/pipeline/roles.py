"""Which output stem of a separation model plays which role in the pipeline.

Models name their stems differently ("No Reverb", "noreverb", "dry"; "No Noise" vs
"No dry"), so each stage lists the labels of the stem it keeps, best match first.
The stems that are not kept are the "rest" (e.g. the instrumental, or the backing vocals).
"""

import re

from ai_cover_maker.core.errors import SeparationError

# stage -> normalized labels of the stem to keep (lowercase, letters and digits only)
KEEP_LABELS = {
    "vocals": ("vocals", "vocal", "voice", "voices"),
    "karaoke": ("vocals", "leadvocals", "lead", "karaoke", "vocal"),
    "dereverb": ("noreverb", "dry", "norev", "dereverb"),
    "deecho": ("noecho", "dry", "noreverb", "deecho"),
    "denoise": ("nonoise", "dry", "denoised", "clean"),
}


def normalize_label(label: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (label or "").lower())


def split_stems(stage: str, stems: dict[str, str]) -> tuple[str, list[str]]:
    """Return ``(kept_file, other_files)`` from ``{label: file}`` of a separation."""
    by_label = {normalize_label(label): path for label, path in stems.items()}
    for wanted in KEEP_LABELS[stage]:
        if wanted in by_label:
            kept = by_label[wanted]
            return kept, [p for p in stems.values() if p != kept]
    raise SeparationError(
        f"Could not tell which stem to keep for the {stage} step from the model outputs "
        f"({', '.join(stems) or 'none'}). Pick another model for this step"
    )
