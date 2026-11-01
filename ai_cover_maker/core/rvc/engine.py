"""RVC voice conversion through the vendored Applio inference code (third_party/applio).

Applio is not a package: it imports itself as ``rvc`` and reads its settings and
weights from environment variables, so ``_setup_applio()`` must run before the first
``import rvc``. Everything Applio writes ends up under ``models/`` and ``user_data/``.
"""

import logging
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ai_cover_maker import paths
from ai_cover_maker.core.device import gpu_lock
from ai_cover_maker.core.errors import ConversionError
from ai_cover_maker.core.rvc import assets
from ai_cover_maker.core.rvc.voices import Voice

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

F0_METHODS = ["rmvpe", "fcpe", "crepe", "swift"]
EMBEDDERS = [*assets.EMBEDDERS, "custom"]
CLEANUP_MODES = ["off", "light", "speech"]


@dataclass(frozen=True)
class RvcParams:
    """Applio ``VoiceConverter.convert_audio`` options exposed in the UI."""

    pitch: int = 0  # semitones
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    volume_envelope: float = 0.25  # 1 = keep the converted loudness, 0 = follow the input's (AICoverGen: 0.25)
    protect: float = 0.33  # protect consonants/breaths; 0.5 disables it
    f0_autotune: bool = False
    f0_autotune_strength: float = 1.0
    proposed_pitch: bool = False  # shift to the model's range automatically
    proposed_pitch_threshold: float = 155.0
    embedder_model: str = "contentvec"
    embedder_model_custom: str | None = None
    formant_shifting: bool = False
    formant_qfrency: float = 1.0
    formant_timbre: float = 1.0
    cleanup_mode: str = "off"
    cleanup_strength: float = 12
    sid: int = 0

    def validate(self) -> None:
        if self.f0_method not in F0_METHODS:
            raise ConversionError(f"Unknown pitch extraction method: {self.f0_method}")
        if self.embedder_model not in EMBEDDERS:
            raise ConversionError(f"Unknown embedder model: {self.embedder_model}")
        if self.embedder_model == "custom" and not (self.embedder_model_custom and Path(self.embedder_model_custom).is_dir()):
            raise ConversionError("The custom embedder must be a folder with pytorch_model.bin and config.json")
        if self.cleanup_mode not in CLEANUP_MODES:
            raise ConversionError(f"Unknown cleanup mode: {self.cleanup_mode}")


_setup_done = False


def _setup_applio() -> None:
    global _setup_done
    if _setup_done:
        return
    os.environ.setdefault("APPLIO_ROOT", str(paths.APPLIO_DIR))
    os.environ.setdefault("APPLIO_MODELS_DIR", str(paths.RVC_ASSETS_DIR))
    os.environ.setdefault("APPLIO_CONFIG_FILE", str(paths.APPLIO_CONFIG_FILE))
    os.environ.setdefault("APPLIO_LOGS_DIR", str(paths.VOICE_MODELS_DIR))
    if str(paths.APPLIO_DIR) not in sys.path:
        sys.path.insert(0, str(paths.APPLIO_DIR))
    _setup_done = True


def import_applio(module: str):
    """Import an Applio module (e.g. ``rvc.infer.infer``) with the environment set up."""
    import importlib

    _setup_applio()
    return importlib.import_module(module)


def _converter_class():
    return import_applio("rvc.infer.infer").VoiceConverter


class RvcEngine:
    """Keeps one ``VoiceConverter`` alive: Applio caches the loaded voice and embedder in it."""

    def __init__(self, factory=None):
        self._factory = factory
        self._converter = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._converter is None:
                self._converter = (self._factory or _converter_class())()
            return self._converter

    def convert(self, input_path: str | Path, output_path: str | Path, voice: Voice, params: RvcParams = RvcParams(),
                progress: ProgressFn = None) -> Path:
        """Convert ``input_path`` with ``voice`` into the WAV ``output_path``."""
        params.validate()
        input_path, output_path = Path(input_path), Path(output_path)
        if not input_path.is_file():
            raise ConversionError(f"Input audio not found: {input_path}")
        if not voice.model_path.is_file():
            raise ConversionError(f"Voice model not found: {voice.model_path}")

        assets.ensure_assets(params.f0_method, params.embedder_model, progress=progress)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_name(f".{output_path.stem}.part.wav")
        kwargs = asdict(params)
        if progress:
            progress(0.0, f"Converting with {voice.name}...")
        try:
            with gpu_lock:
                self._get().convert_audio(
                    audio_input_path=str(input_path),
                    audio_output_path=str(tmp),
                    model_path=str(voice.model_path),
                    index_path=str(voice.index_path or ""),
                    export_format="WAV",
                    **kwargs,
                )
            if not tmp.is_file():
                raise ConversionError("RVC produced no output, check the console for details")
            os.replace(tmp, output_path)
        except ConversionError:
            raise
        except Exception as e:
            self.unload()
            raise ConversionError(f"Voice conversion failed: {str(e) or type(e).__name__}") from e
        finally:
            tmp.unlink(missing_ok=True)
        return output_path

    def unload(self) -> None:
        """Free the voice and embedder (e.g. before a separation model needs the VRAM)."""
        with self._lock:
            converter, self._converter = self._converter, None
        if converter is not None:
            try:
                converter.cleanup_model()
            except Exception as e:  # best effort
                logger.debug("cleanup_model failed: %s", e)
            del converter
            _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


rvc_engine = RvcEngine()
