from pathlib import Path

import numpy as np
from scipy.io import wavfile

CLICK_HZ = 2000.0
CLICK_SECONDS = 0.06
CLICK_DECAY_SECONDS = 0.02


def click(sample_rate: int) -> np.ndarray:
    t = np.arange(int(CLICK_SECONDS * sample_rate)) / sample_rate
    return np.sin(2.0 * np.pi * CLICK_HZ * t) * np.exp(-t / CLICK_DECAY_SECONDS)


def sonify(
    signal: np.ndarray,
    times: np.ndarray,
    sample_rate: int,
    track_gain_db: float = -12.0,
    click_gain: float = 0.9,
) -> np.ndarray:
    """Mix a click at each time (seconds) over the attenuated signal."""
    out = signal.astype(np.float32) * 10.0 ** (track_gain_db / 20.0)
    burst = (click_gain * click(sample_rate)).astype(np.float32)
    for time in times:
        start = round(time * sample_rate)
        if not 0 <= start < len(out):
            continue
        stop = min(len(out), start + len(burst))
        out[start:stop] += burst[: stop - start]
    peak = float(np.max(np.abs(out), initial=0.0))
    return out / peak if peak > 1.0 else out


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    wavfile.write(path, sample_rate, signal.astype(np.float32))
