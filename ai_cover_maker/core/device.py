"""Compute device detection. torch is imported lazily because it's slow to load."""

import functools
import threading


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def use_autocast() -> bool:
    return get_device() == "cuda"


# Held around every model load and inference (separation and RVC) so two GPU jobs never
# overlap. Reentrant, so the cover pipeline can hold it across stages that also take it.
gpu_lock = threading.RLock()
