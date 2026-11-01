"""Glue between Gradio event handlers and the core services."""

import contextvars
import functools
import inspect
import logging
import queue
import threading

import gradio as gr

from ai_cover_maker.core.errors import AppError
from ai_cover_maker.i18n import i18n

logger = logging.getLogger(__name__)

# Concurrency group shared by every GPU-heavy event so two separations never overlap.
GPU_CONCURRENCY_ID = "separation"


def progress_callback(progress: gr.Progress):
    """Adapt ``gr.Progress`` to the core's ``(fraction, description)`` callback."""
    def report(fraction: float, desc: str) -> None:
        progress(fraction, desc=desc)
    return report


def user_errors(func):
    """Show failures as Gradio error toasts instead of raw tracebacks.

    Works for plain functions and generator handlers.
    """
    def to_gr_error(e: Exception) -> gr.Error:
        if isinstance(e, gr.Error):
            return e
        if isinstance(e, AppError):
            logger.warning("%s", e, exc_info=e.__cause__ is not None)
            return gr.Error(i18n(str(e)))  # some messages are i18n keys; the rest pass through
        logger.exception("Unexpected error in %s", func.__name__)
        return gr.Error(f"{i18n('An unexpected error occurred, check the console for details')}: {e}")

    if inspect.isgeneratorfunction(func):
        @functools.wraps(func)
        def gen_wrapper(*args, **kwargs):
            try:
                yield from func(*args, **kwargs)
            except Exception as e:
                raise to_gr_error(e) from e
        return gen_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise to_gr_error(e) from e
    return wrapper


def run_streaming(func, render=lambda message: message, progress=None, poll: float = 0.2):
    """Run ``func(emit, progress)`` in a worker thread and yield ``render(value)`` for every value it emits.

    Use it with ``result = yield from run_streaming(...)``: the return value is ``func``'s, and
    its exception is raised here. Gradio's context is copied to the thread, so ``gr.Progress``
    keeps working inside ``func``. Every yield hides the progress bars of the event's outputs
    until the next progress report, so the last report of ``progress`` is sent again after it.
    """
    messages = queue.Queue()
    outcome = {}
    last = []

    def report(fraction: float, desc: str) -> None:
        last[:] = [fraction, desc]
        if progress:
            progress(fraction, desc)

    def target():
        try:
            outcome["value"] = func(messages.put, report)
        except BaseException as e:
            outcome["error"] = e

    thread = threading.Thread(target=contextvars.copy_context().run, args=(target,), daemon=True)
    thread.start()
    while thread.is_alive() or not messages.empty():
        try:
            yield render(messages.get(timeout=poll))
        except queue.Empty:
            continue
        if progress and last:
            progress(*last)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def validations(*checks) -> dict:
    """Validator result from ``(component, is_valid, message)`` checks.

    Validators run before the event is queued, so a missing input is flagged on its
    component right away instead of after waiting for the GPU.
    """
    return {component: gr.validate(bool(is_valid), i18n(message)) for component, is_valid, message in checks}


def filled(value) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def require(value, message: str):
    """Raise a translated Gradio error when a required input is missing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise gr.Error(i18n(message))
    return value
