"""UI smoke tests: build the Blocks app without launching it and drive handlers directly."""

import pytest

gr = pytest.importorskip("gradio")

from ai_cover_maker import paths  # noqa: E402
from ai_cover_maker.config.settings import CoverSettings  # noqa: E402
from ai_cover_maker.config.store import ConfigStore  # noqa: E402
from ai_cover_maker.core.models.catalog import ModelCatalog  # noqa: E402
from ai_cover_maker.core.separation import runtime  # noqa: E402
from ai_cover_maker.core.separation.runtime import separator_cache  # noqa: E402
from ai_cover_maker.ui.app import create_app  # noqa: E402
from ai_cover_maker.ui.components import model_choices  # noqa: E402
from ai_cover_maker.ui import a11y  # noqa: E402
from ai_cover_maker.ui.app import launch_kwargs  # noqa: E402
from ai_cover_maker.ui.context import AppContext  # noqa: E402
from tests.test_separation import FakeSeparator  # noqa: E402

class NoProgress:
    def __call__(self, *args, **kwargs):
        pass


@pytest.fixture
def ctx(tmp_root, monkeypatch):
    monkeypatch.setattr(runtime, "use_autocast", lambda: False)
    monkeypatch.setattr(runtime, "_separator_class", lambda: FakeSeparator)
    separator_cache.clear()
    FakeSeparator.instances.clear()
    config = ConfigStore(tmp_root / "assets" / "config.default.json", tmp_root / "user_data" / "config.json")
    config.set("theme.class", None)  # avoid fetching a Hub theme
    return AppContext(config, ModelCatalog.from_file(tmp_root / "assets" / "models.json"),
                      CoverSettings(tmp_root / "user_data" / "cover_settings.json"))


@pytest.fixture
def app(ctx):
    return create_app(ctx)


def find_fn(app, name, nth=0):
    fns = [f for f in app.fns.values() if getattr(f, "name", None) == name or getattr(f, "api_name", None) == name]
    return fns[nth]


def kwargs_of(fn, **overrides):
    """Every input of an inputs_kwargs event at its initial value, by parameter name."""
    values = {name: comp.value for name, comp in zip(fn.input_keyword_names, fn.inputs)}
    values.update(overrides)
    return values


def invalid(result: dict) -> dict:
    """{component label: message} for the failed checks of a validator."""
    return {comp.label: v["message"] for comp, v in result.items() if not v["is_valid"]}


def test_tabs(app):
    labels = [b.label for b in app.blocks.values() if type(b).__name__ == "Tab"]
    assert labels == ["Cover", "Voices", "Themes", "Settings", "Credits"]


def test_telemetry_disabled(app):
    assert app.analytics_enabled is False


def test_temp_cache_is_cleaned(app):
    assert app.delete_cache == (86400, 86400)


def test_batch_can_be_stopped(app):
    batch = find_fn(app, "make_covers_batch")
    cancellers = [f for f in app.fns.values() if f.fn is None and batch._id in (f.cancels or [])]
    assert len(cancellers) == 1


def test_recommended_models_come_first_with_their_stem(ctx):
    choices = model_choices(ctx.catalog, ctx.catalog.by_arch("roformer"))
    assert choices[:2] == [
        ("BS Roformer | Instrumental Resurrection by unwa (Instrumental ⭐)",
         "BS Roformer | Instrumental Resurrection by unwa"),
        ("BS Roformer SW by jarredou (Vocals ⭐)", "BS Roformer SW by jarredou"),
    ]
    # The values stay the model ids, so saved settings and the API are unchanged
    assert sorted(value for _, value in choices) == sorted(ctx.catalog.ids("roformer"))
    assert sum("⭐" in label for label, _ in choices) == 5


def test_page_and_gradio_language_follow_the_app_language():
    assert a11y.page_language("pt_BR") == "pt-BR"
    locales = [a11y.gradio_locale(code) for code in ("es_ES", "pt_BR", "zh_CN", "it_IT", "vi_VN")]
    assert locales == ["es", "pt-BR", "zh-CN", "en", "en"]  # Gradio has no Italian or Vietnamese texts
    head = a11y.head_html("es_ES")
    assert head.startswith("<script>") and head.endswith("</script>")
    assert '"page_lang": "es-ES"' in head and "__CONFIG__" not in head


def test_launch_adds_the_accessibility_fixes(ctx):
    kwargs = launch_kwargs(ctx)
    assert kwargs["head"].startswith("<script>")
    assert ":focus-visible" in kwargs["css"]
    # Gradio's progress announcement reads this key, which no Gradio language defines
    assert kwargs["i18n"].translations_dict == {"en": {"common.complete": "Complete"}}
    assert a11y.gradio_i18n("pt_BR").translations_dict.keys() == {"pt-BR"}


def test_announcements():
    assert a11y.announce("Hi")["id"] != a11y.announce("Hi")["id"]  # repeated messages are read again


LABELLED = ("Textbox", "Dropdown", "Slider", "Checkbox", "CheckboxGroup", "Radio", "Number", "Audio", "File")


def test_every_control_has_a_label_and_buttons_are_distinct(app):
    # Screen readers announce controls by their label: none may be empty, and two buttons in
    # the same tab may not share a name (e.g. the single and batch "Separate!" buttons)
    unlabelled = [f"{type(b).__name__} #{b._id}" for b in app.blocks.values()
                  if type(b).__name__ in LABELLED and not str(b.label or "").strip()]
    assert not unlabelled

    def buttons(block):
        for child in getattr(block, "children", []):
            if type(child).__name__ == "Tab":
                continue  # nested tabs are checked on their own
            if type(child).__name__ == "Button":
                yield str(child.value)
            elif type(child).__name__ == "DownloadButton":
                yield str(child.label)  # its text; the value is the file
            yield from buttons(child)

    for tab in (b for b in app.blocks.values() if type(b).__name__ == "Tab"):
        names = list(buttons(tab))
        assert len(names) == len(set(names)), (tab.label, names)



def test_every_tab_has_a_heading_for_screen_readers(app):
    tabs = [b for b in app.blocks.values() if type(b).__name__ == "Tab"]
    for tab in tabs:
        headings = [c for c in tab.children if type(c).__name__ == "HTML" and "uvr-sr-only" in (c.elem_classes or [])]
        assert headings and f"<h2>{tab.label}</h2>" in headings[0].value, tab.label


def test_language_selector_shows_names(app):
    selector = next(b for b in app.blocks.values() if type(b).__name__ == "Dropdown" and b.label == "Language")
    choices = dict((value, label) for label, value in selector.choices)
    assert choices["es_ES"] == "Español (es_ES)"
    assert choices["pt_BR"] == "Português (Brasil) (pt_BR)"


# -- Cover and Voices tabs ---------------------------------------------

@pytest.fixture
def voices(tmp_root, monkeypatch):
    folder = tmp_root / "models" / "rvc" / "Singer"
    folder.mkdir(parents=True)
    (folder / "singer.pth").write_bytes(b"x")
    monkeypatch.setattr(paths, "VOICE_MODELS_DIR", tmp_root / "models" / "rvc")
    return tmp_root / "models" / "rvc"


def test_cover_builds_the_pipeline_options(voices, ctx, tmp_root, monkeypatch):
    from ai_cover_maker.core.pipeline.runner import CoverResult
    from ai_cover_maker.ui.tabs import cover as cover_tab

    seen = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def run(self, song, options, progress=None, on_log=None):
            seen["options"] = options
            on_log(["Separating..."])
            return CoverResult(tmp_root / "cover.wav", stems={"instrumental": tmp_root / "inst.wav"}, log=["done"])

    monkeypatch.setattr(cover_tab, "CoverPipeline", FakePipeline)
    app = create_app(ctx)
    fn = find_fn(app, "make_cover")
    assert invalid(fn.validator(**kwargs_of(fn, voice=None))) == {
        "Voice model": "Please select a voice model", "Song": "Please provide an input audio"}

    kwargs = kwargs_of(fn, audio="song.wav", pitch=12.0, backing_pitch=-2.0, key_shift=1.0, reverb=True,
                       karaoke_model="", backing_voice="Singer")
    updates = list(fn.fn(**kwargs, progress=NoProgress()))
    assert updates[0][3] == "Separating..." and all(isinstance(u, type(gr.skip())) for u in updates[0][:3])
    cover, inst, backing, log = updates[-1]
    options = seen["options"]
    assert options.voice == "Singer" and options.vocals_model == "MelBand Roformer | Vocals by Kimberley Jensen"
    assert options.karaoke_model is None and options.dereverb_model == "MelBand Roformer | De-Reverb by anvuew"
    assert (options.rvc.pitch, options.backing_rvc.pitch, options.key_shift) == (12, -2, 1)
    assert isinstance(options.rvc.pitch, int) and options.effects.reverb is True
    assert cover.endswith("cover.wav") and backing["visible"] is False and log == "done"
    # Separation parameters and precision are always the defaults
    assert set(seen["init"]) == {"catalog", "outputs_dir"}
    # The progress bars are on the audio players, never on the log
    players = [cover_component(app, label) for label in ("Cover", "Instrumental", "Backing vocals")]
    assert fn.show_progress_on == players
    # ...over empty players: the browser clears them right before the cover starts
    clear = next(f for f in app.fns.values() if f.fn is None and f.js and "null" in f.js)
    assert [o._id for o in clear.outputs] == [p._id for p in players]


def test_cover_presets_fill_the_steps(voices, ctx):
    app = create_app(ctx)
    preset = next(f for f in app.fns.values() if getattr(f.fn, "__name__", "") == "apply_preset")
    info, vocals, karaoke, dereverb, deecho, denoise = preset.fn("Fast")
    assert vocals == "MelBand Roformer | Vocals by Kimberley Jensen" and karaoke == "" and dereverb == "UVR-DeEcho-DeReverb.pth"
    assert "quickest" in info["info"]


def test_voices_tab_upload_and_delete(voices, ctx, tmp_path):
    app = create_app(ctx)
    upload = find_fn(app, "upload_voice")
    model = tmp_path / "other.pth"
    model.write_bytes(b"x")
    table, dropdown, status = upload.fn([str(model)], "Other")
    assert ["Other", "other.pth", "—"] in table and "Other" in dropdown["choices"]
    assert (voices / "Other" / "other.pth").is_file()

    with pytest.raises(gr.Error):
        upload.fn([str(model)], "")
    table, _, _ = find_fn(app, "delete_voice").fn("Other")
    assert [row[0] for row in table] == ["Singer"]


def cover_component(app, label):
    return next(b for b in app.blocks.values() if type(b).__name__ == "Audio" and b.label == label)


def test_cover_top_row_and_button_heights(voices, ctx):
    app = create_app(ctx)

    def items(row):
        # Gradio wraps consecutive inputs of a row in a Form
        for child in row.children:
            yield from (child.children if type(child).__name__ == "Form" else [child])

    rows = [list(items(b)) for b in app.blocks.values() if type(b).__name__ == "Row"]
    top = next(r for r in rows if any(getattr(c, "label", None) == "Voice model" for c in r))
    assert [c.value if type(c).__name__ == "Button" else c.label for c in top] == [
        "Voice model", "Refresh voice models", "Select the output format"]
    # Buttons next to a dropdown or textbox are as tall as it
    for row, block in zip(rows, (b for b in app.blocks.values() if type(b).__name__ == "Row")):
        kinds = {type(c).__name__ for c in row}
        if "Button" in kinds and kinds & {"Dropdown", "Textbox"}:
            assert block.equal_height, [getattr(c, "label", None) or c.value for c in row]


def test_run_streaming_repeats_the_progress_after_each_yield():
    from ai_cover_maker.ui.adapters import run_streaming

    reports = []

    def work(emit, report):
        report(0.5, "Converting")
        emit("Converting...")
        return "result"

    def handler():
        result = yield from run_streaming(work, progress=lambda fraction, desc: reports.append(fraction))
        yield result

    # A yield hides the progress bars until the next report, so the last one is sent again
    assert list(handler()) == ["Converting...", "result"]
    assert reports == [0.5, 0.5]


def test_cover_settings_are_saved_loaded_and_reset(voices, ctx):
    from ai_cover_maker.core.pipeline.options import load_presets
    from ai_cover_maker.ui.tabs.cover import default_settings

    app = create_app(ctx)
    save = next(f for f in app.fns.values() if getattr(f.fn, "__name__", "") == "save_settings")
    names = [getattr(c, "label", None) for c in save.inputs]
    values = [c.value for c in save.inputs]
    values[names.index("Search feature ratio")] = 0.3
    values[names.index("Reverb")] = True
    values[names.index("Preset")] = "Fast"
    save.fn(*values)
    saved = ctx.cover_settings.load()
    # Every saved setting has a default to go back to
    assert set(saved) == set(default_settings(load_presets()))
    assert (saved["index_rate"], saved["reverb"], saved["preset"]) == (0.3, True, "Fast")

    # Loaded the next time the app starts; values that are no longer valid fall back to the default
    ctx.cover_settings.save({**saved, "voice": "Deleted voice", "pitch": "high", "unknown": 1})
    new_app = create_app(ctx)
    inputs = next(f for f in new_app.fns.values() if getattr(f.fn, "__name__", "") == "save_settings").inputs
    reloaded = {c.label: c.value for c in inputs}
    assert (reloaded["Search feature ratio"], reloaded["Reverb"], reloaded["Preset"]) == (0.3, True, "Fast")
    assert reloaded["Voice model"] == "Singer" and reloaded["Pitch"] == 0

    reset = next(f for f in app.fns.values() if getattr(f.fn, "__name__", "") == "reset_settings")
    defaults = dict(zip([c.label for c in reset.outputs], reset.fn()))
    assert defaults["Search feature ratio"] == 0.75 and defaults["Preset"] == "Balanced"
    assert ctx.cover_settings.load() == {} and not ctx.cover_settings.file.exists()


def test_effect_settings_are_editable_only_while_the_effect_is_checked(app):
    blocks = {getattr(b, "label", None): b for b in app.blocks.values()}
    room = blocks["Room size"]
    assert blocks["Reverb"].value is True and room.interactive is True
    # High-pass has a single setting: Gradio would take a returned list as that slider's value
    for effect, setting in (("Reverb", room), ("High-pass filter", blocks["Cutoff frequency (Hz)"])):
        toggle = next(f for f in app.fns.values() if f.inputs == [blocks[effect]])
        assert setting in toggle.outputs
        for on in (False, True):
            updates = toggle.fn(on)
            assert set(updates) == set(toggle.outputs)
            assert all(update == {"__type__": "update", "interactive": on} for update in updates.values())
