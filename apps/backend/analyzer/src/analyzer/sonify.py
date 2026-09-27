from pathlib import Path

import numpy as np
from scipy.io import wavfile

CLICK_HZ = 2000.0
CLICK_SECONDS = 0.06
CLICK_DECAY_SECONDS = 0.02
MIN_CLICK_LEVEL = 0.2  # click level at confidence 0


def click(sample_rate: int) -> np.ndarray:
    t = np.arange(int(CLICK_SECONDS * sample_rate)) / sample_rate
    return np.sin(2.0 * np.pi * CLICK_HZ * t) * np.exp(-t / CLICK_DECAY_SECONDS)


def sonify(
    signal: np.ndarray,
    times: np.ndarray,
    sample_rate: int,
    track_gain_db: float = -12.0,
    click_gain: float = 0.9,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Mix a click at each time (seconds) over the attenuated signal.

    With `confidence` (0-1 per time), the click is louder for higher confidence.
    """
    out = signal.astype(np.float32) * 10.0 ** (track_gain_db / 20.0)
    burst = (click_gain * click(sample_rate)).astype(np.float32)
    if confidence is None:
        confidence = np.ones(len(times))
    levels = MIN_CLICK_LEVEL + (1.0 - MIN_CLICK_LEVEL) * np.clip(confidence, 0.0, 1.0)
    for time, level in zip(times, levels):
        start = round(time * sample_rate)
        if not 0 <= start < len(out):
            continue
        stop = min(len(out), start + len(burst))
        out[start:stop] += level * burst[: stop - start]
    peak = float(np.max(np.abs(out), initial=0.0))
    return out / peak if peak > 1.0 else out


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int) -> None:
    wavfile.write(path, sample_rate, signal.astype(np.float32))
