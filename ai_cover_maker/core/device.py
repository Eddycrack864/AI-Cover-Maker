"""Compute device detection. torch is imported lazily because it's slow to load."""

import functools
import threading


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def use_autocast() -> bool:
    return get_device() == "cuda"


LOW_VRAM_GB = 8


@functools.lru_cache(maxsize=1)
def low_vram() -> bool:
    """True on GPUs too small to keep a separation model and an RVC voice loaded together."""
    if get_device() != "cuda":
        return False  # system RAM: keep models warm
    import torch

    return torch.cuda.get_device_properties(0).total_memory <= LOW_VRAM_GB * 1024**3


# Held around every model load and inference (separation and RVC) so two GPU jobs never
# overlap. Reentrant, so the cover pipeline can hold it across stages that also take it.
gpu_lock = threading.RLock()
