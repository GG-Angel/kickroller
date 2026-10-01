"""Audio in and out: decoding, the mid channel, loudness and WAV files."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm
from loguru import logger
from scipy.io import wavfile

SAMPLE_RATE = 44100
TARGET_LUFS = -14.0


def load_mid(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio or video file to the mid channel (L+R)/2 at `sample_rate`.

    ffmpeg does the decoding and resampling. It honors gapless metadata
    (MP3 encoder delay, AAC priming in MP4/M4A), so time zero is the first
    real sample of the track.
    """
    logger.debug("Decoding {path} with ffmpeg", path=path)
    signal = decode_mid(path, sample_rate)
    logger.info(
        "Decoded {name}: {seconds:.1f} s at {rate} Hz",
        name=Path(path).name,
        seconds=len(signal) / sample_rate,
        rate=sample_rate,
    )
    return signal


def decode_mid(
    path: str | Path, sample_rate: int = SAMPLE_RATE, tempo: float = 1.0
) -> np.ndarray:
    """`load_mid` with no logs.

    A `tempo` other than 1 makes the audio faster (above 1) or slower and keeps
    its pitch, with ffmpeg's `atempo` filter (WSOLA). WSOLA copies short
    segments of the input, so a kick attack stays as sharp as in the source.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH; install it to decode audio")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")

    stretch = ["-af", f"atempo={tempo:.6f}"] if tempo != 1.0 else []
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", str(path),
        "-map", "0:a:0",
        *stretch,
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
        logger.warning(
            "Signal is shorter than {block:g} s; loudness is not normalized",
            block=meter.block_size,
        )
        return signal
    loudness = meter.integrated_loudness(signal)
    if not np.isfinite(loudness):
        logger.warning("Signal is silent; loudness is not normalized")
        return signal
    gain_db = target_lufs - loudness
    logger.debug(
        "Loudness {loudness:.1f} LUFS; gain {gain:+.1f} dB to {target:.1f} LUFS",
        loudness=loudness,
        gain=gain_db,
        target=target_lufs,
    )
    return signal * 10.0 ** (gain_db / 20.0)


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    wavfile.write(path, sample_rate, signal.astype(np.float32))
