# 07 — Discord Rich Presence (done, needs an Application ID)

The user wants to keep Rich Presence, so `ai_cover_maker/integrations/discord_presence.py` is ported
from UVR5-UI.

- Activities: `cover`, `convert` and the five separation `ArchSpec` ids, each with a "Batch" variant.
- Wired into the Cover, Voice conversion and Separation handlers through `presence.track()`.
- Settings has a "Discord Rich Presence" checkbox (`discord_presence`, default `true`); it applies after
  a restart.
- `CLIENT_ID` is empty until the user creates the "AI Cover Maker" app in the Discord Developer
  Portal. With an empty id the presence logs a message and stays off.
- Image assets the Discord app needs (Rich Presence → Art Assets): `logo`, `idling`, `cover`,
  `convert`, `roformer`, `mdxc`, `mdxnet`, `vrarch`, `demucs`.
- Tests: `tests/test_discord_presence.py` (fake pypresence).
