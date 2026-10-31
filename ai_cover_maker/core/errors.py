"""Domain errors. The UI layer turns these into user-facing messages."""


class AppError(Exception):
    """Base class for expected, user-reportable failures."""


class SeparationError(AppError):
    pass


class DownloadError(AppError):
    pass


class ConversionError(AppError):
    """RVC voice conversion failed."""


class MixError(AppError):
    pass
