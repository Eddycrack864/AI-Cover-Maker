"""Optional Discord Rich Presence. Never raises: if Discord isn't running it disables itself."""

import contextlib
import datetime as dt
import logging
import threading

logger = logging.getLogger(__name__)

# Application ID of the "AI Cover Maker" app in the Discord Developer Portal. Empty disables the presence.
CLIENT_ID = "1566198256130719857"
IDLE = "Idling"

# activity id -> (small_image asset, label shown in Discord). The app has the "logo", "idling" and
# "cover" image assets; a new activity needs its image uploaded there first.
ACTIVITIES = {
    "cover": ("cover", "AI Cover"),
}


class RichPresenceManager:
    def __init__(self, client_id: str = CLIENT_ID):
        self.client_id = client_id
        self.rpc = None
        self.running = False
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return
        if not self.client_id:
            logger.info("No Discord application ID set. Rich Presence will be disabled.")
            return
        try:
            from pypresence import Presence
            from pypresence.exceptions import DiscordNotFound, InvalidPipe
        except ImportError:
            logger.info("pypresence is not installed. Rich Presence will be disabled.")
            return

        try:
            self.rpc = Presence(self.client_id)
            self.rpc.connect()
            self.running = True
            logger.info("Discord Rich Presence connected successfully")
            self._update(IDLE, "idling", IDLE)
        except (DiscordNotFound, InvalidPipe):
            logger.info("Discord is not running. Rich Presence will be disabled.")
            self.rpc = None
        except Exception as e:
            logger.warning("An error occurred connecting to Discord: %s", e)
            self.rpc = None

    def _update(self, state: str, small_image: str, small_text: str) -> None:
        if not (self.rpc and self.running):
            return
        try:
            self.rpc.update(
                state=state,
                details="AI Cover Maker",
                buttons=[{"label": "Download", "url": "https://github.com/Eddycrack864/AI-Cover-Maker"}],
                large_image="logo",
                large_text="Making AI covers with AI Cover Maker",
                small_image=small_image,
                small_text=small_text,
                start=dt.datetime.now().timestamp(),
            )
        except Exception as e:
            logger.warning("Error updating Discord presence: %s", e)
            self.stop()

    def set_activity(self, activity: str | None, batch: bool = False) -> None:
        with self.lock:
            if activity is None or activity not in ACTIVITIES:
                self._update(IDLE, "idling", IDLE)
                return
            image, label = ACTIVITIES[activity]
            self._update(f"{'Batch ' if batch else ''}{label}", image, label)

    @contextlib.contextmanager
    def track(self, activity: str, batch: bool = False):
        """Show the activity while the block runs, then go back to idling."""
        self.set_activity(activity, batch)
        try:
            yield
        finally:
            self.set_activity(None)

    def stop(self) -> None:
        self.running = False
        if self.rpc:
            try:
                self.rpc.close()
            except Exception:
                pass
        self.rpc = None


presence = RichPresenceManager()
