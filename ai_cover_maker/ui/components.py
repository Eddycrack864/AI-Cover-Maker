"""Reusable Gradio widget factories. Must be called inside a ``gr.Blocks`` context."""

import gradio as gr

from ai_cover_maker.core.models.catalog import Model, ModelCatalog
from ai_cover_maker.i18n import i18n


def model_label(catalog: ModelCatalog, model: Model, prefix: str = "") -> str:
    """Dropdown label of a model: its id, plus e.g. ``(Vocals ⭐)`` when it is a recommended model."""
    stems = catalog.recommended_stems(model)
    return f"{prefix}{model.id} ({', '.join(stems)} ⭐)" if stems else prefix + model.id


def model_choices(catalog: ModelCatalog, models: list[Model]) -> list[tuple[str, str]]:
    """``(label, id)`` dropdown choices, recommended models first."""
    return [(model_label(catalog, m), m.id) for m in catalog.recommended_first(models)]


def single_line_textbox(label: str, placeholder: str | None, value: str = "", visible: bool = True):
    return gr.Textbox(label=label, placeholder=placeholder, value=value, lines=1, max_lines=1, interactive=True,
                      visible=visible)


def output_log(label: str = None):
    return gr.Textbox(label=label or i18n("Output information"), lines=5, interactive=False)
