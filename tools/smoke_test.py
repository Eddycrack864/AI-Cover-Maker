"""End-to-end dependency smoke test: separate -> RVC -> mix.

Usage:
    python tools/smoke_test.py --input song.wav --voice models/rvc/<name>/<name>.pth [--index X.index]
                               [--device cuda|cpu] [--sep-model vocals_mel_band_roformer.ckpt]

Standalone on purpose (does not import ai_cover_maker) so it validates the
dependency stack before/independently of the app code. Writes to .local/smoke/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLIO = ROOT / "third_party" / "applio"
RVC_ASSETS = ROOT / "models" / "rvc_assets"
HF_RESOURCES = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"


def setup_applio_env() -> None:
    os.environ.setdefault("APPLIO_ROOT", str(APPLIO))
    os.environ.setdefault("APPLIO_MODELS_DIR", str(RVC_ASSETS))
    os.environ.setdefault("APPLIO_CONFIG_FILE", str(ROOT / "user_data" / "applio_config.json"))
    os.environ.setdefault("APPLIO_LOGS_DIR", str(ROOT / "models" / "rvc"))
    if str(APPLIO) not in sys.path:
        sys.path.insert(0, str(APPLIO))


def ensure_asset(rel: str) -> None:
    dst = RVC_ASSETS / rel
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {rel} ...")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(f"{HF_RESOURCES}/{rel}", tmp)
    tmp.replace(dst)


class Timer:
    def __init__(self, label: str, device: str):
        self.label, self.device = label, device

    def __enter__(self):
        import torch

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        import torch

        dt = time.perf_counter() - self.t0
        vram = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB" if self.device == "cuda" else ""
        print(f"[{self.label}] {dt:.1f}s{vram}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--voice", type=Path, required=True)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--sep-model", default="vocals_mel_band_roformer.ckpt")
    p.add_argument("--pitch", type=int, default=0)
    args = p.parse_args()

    setup_applio_env()
    import numpy as np
    import soundfile as sf
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # keep Applio's Config and audio-separator on CPU
    print(f"torch {torch.__version__} | device {device}")
    out = ROOT / ".local" / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Separation
    from audio_separator.separator import Separator

    with Timer("separation", device):
        sep = Separator(output_dir=str(out), model_file_dir=str(ROOT / "models" / "separation"), output_format="WAV")
        sep.load_model(model_filename=args.sep_model)
        stems = [out / s for s in sep.separate(str(args.input))]
    vocals = next(s for s in stems if "(vocals)" in s.name.lower())
    inst = next(s for s in stems if s != vocals)
    del sep
    print("stems:", [s.name for s in stems])

    # 2. RVC
    for rel in ("predictors/rmvpe.pt", "predictors/fcpe.pt"):
        ensure_asset(rel)
    from rvc.infer.infer import VoiceConverter

    converted = out / "vocals_rvc.wav"
    with Timer("rvc", device):
        VoiceConverter().convert_audio(
            audio_input_path=str(vocals),
            audio_output_path=str(converted),
            model_path=str(args.voice),
            index_path=str(args.index or ""),
            pitch=args.pitch,
            f0_method="rmvpe",
            index_rate=0.5,
            volume_envelope=1.0,
            protect=0.33,
            embedder_model="contentvec",
            export_format="WAV",
        )

    # 3. Mix (pedalboard reverb + limiter on vocals, align sample rate/length)
    import librosa
    from pedalboard import Limiter, Pedalboard, Reverb

    with Timer("mix", "cpu"):
        inst_audio, sr = sf.read(inst, always_2d=True)
        voc, vsr = sf.read(converted, always_2d=True)
        if vsr != sr:
            voc = librosa.resample(voc.T, orig_sr=vsr, target_sr=sr).T
        if voc.shape[1] != inst_audio.shape[1]:
            voc = np.repeat(voc[:, :1], inst_audio.shape[1], axis=1)
        n = max(len(voc), len(inst_audio))
        voc = np.pad(voc, ((0, n - len(voc)), (0, 0)))
        inst_audio = np.pad(inst_audio, ((0, n - len(inst_audio)), (0, 0)))
        voc = Pedalboard([Reverb(room_size=0.3, wet_level=0.15)])(voc.T.astype(np.float32), sr).T
        mix = Pedalboard([Limiter(threshold_db=-1.0)])((voc + inst_audio * 10 ** (-3 / 20)).T.astype(np.float32), sr).T
        final = out / "cover.wav"
        sf.write(final, mix, sr)
    print(f"OK -> {final} (peak {20 * np.log10(np.abs(mix).max() + 1e-9):.2f} dBFS, {n / sr:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
