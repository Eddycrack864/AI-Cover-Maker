"""Discord Rich Presence never raises and stays off without an application ID."""

import sys
import types

from ai_cover_maker.integrations import discord_presence
from ai_cover_maker.integrations.discord_presence import RichPresenceManager


class FakePresence:
    def __init__(self, client_id):
        self.client_id = client_id
        self.updates = []
        self.closed = False

    def connect(self):
        pass

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def close(self):
        self.closed = True


def _fake_pypresence(monkeypatch):
    module = types.ModuleType("pypresence")
    module.Presence = FakePresence
    exceptions = types.ModuleType("pypresence.exceptions")
    exceptions.DiscordNotFound = type("DiscordNotFound", (Exception,), {})
    exceptions.InvalidPipe = type("InvalidPipe", (Exception,), {})
    monkeypatch.setitem(sys.modules, "pypresence", module)
    monkeypatch.setitem(sys.modules, "pypresence.exceptions", exceptions)


def test_without_client_id_it_stays_off(monkeypatch):
    _fake_pypresence(monkeypatch)
    manager = RichPresenceManager(client_id="")
    manager.start()
    assert not manager.running
    with manager.track("cover"):
        pass  # no-op, no error


def test_tracks_activity_then_idles(monkeypatch):
    _fake_pypresence(monkeypatch)
    manager = RichPresenceManager(client_id="123")
    manager.start()
    assert manager.running
    with manager.track("cover", batch=True):
        assert manager.rpc.updates[-1]["state"] == "Batch AI Cover"
        assert manager.rpc.updates[-1]["small_image"] == "cover"
    assert manager.rpc.updates[-1]["state"] == discord_presence.IDLE
    rpc = manager.rpc
    manager.stop()
    assert rpc.closed and manager.rpc is None


def test_every_separation_tab_has_an_activity():
    from ai_cover_maker.core.separation.specs import ARCHITECTURES

    assert {spec.id for spec in ARCHITECTURES} <= set(discord_presence.ACTIVITIES)
