import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm

SAMPLE_RATE = 44100
TARGET_LUFS = -14.0


def load_mid(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio or video file to the mid channel (L+R)/2 at `sample_rate`.

    ffmpeg does the decoding and resampling. It honors gapless metadata
    (MP3 encoder delay, AAC priming in MP4/M4A), so time zero is the first
    real sample of the track.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH; install it to decode audio")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", str(path),
        "-map", "0:a:0",
        "-ac", "2",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        message = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {message}")

    stereo = np.frombuffer(proc.stdout, dtype="<f4").reshape(-1, 2)
    return stereo.mean(axis=1)


def normalize_loudness(
    signal: np.ndarray, sample_rate: int = SAMPLE_RATE, target_lufs: float = TARGET_LUFS
) -> np.ndarray:
    """Scale `signal` to an integrated loudness of `target_lufs` (ITU-R BS.1770)."""
    meter = pyloudnorm.Meter(sample_rate)
    if len(signal) < int(meter.block_size * sample_rate):
        return signal
    loudness = meter.integrated_loudness(signal)
    if not np.isfinite(loudness):
        return signal
    return signal * 10.0 ** ((target_lufs - loudness) / 20.0)
