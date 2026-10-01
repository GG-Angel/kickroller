"""Click tracks: the audio with a click at each kick, to check kicks by ear."""

import numpy as np

from analyzer.audio.io import db_to_gain

CLICK_HZ = 2000.0
CLICK_SECONDS = 0.06
CLICK_DECAY_SECONDS = 0.02
CLICK_GAIN = 0.9  # the peak of a click at confidence 1
MIN_CLICK_LEVEL = 0.2  # click level at confidence 0, relative to confidence 1
TRACK_GAIN_DB = -12.0  # the track is quieter, so the clicks are clear


def make_click(sample_rate: int) -> np.ndarray:
    """A short sine burst with an exponential decay."""
    t = np.arange(int(CLICK_SECONDS * sample_rate)) / sample_rate
    return np.sin(2.0 * np.pi * CLICK_HZ * t) * np.exp(-t / CLICK_DECAY_SECONDS)


def mix_clicks(
    signal: np.ndarray,
    times: np.ndarray,
    sample_rate: int,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Mix a click at each time (seconds) over the attenuated signal.

    With `confidence` (0-1 per time), the click is louder for higher confidence.
    """
    out = signal.astype(np.float32) * db_to_gain(TRACK_GAIN_DB)
    burst = (CLICK_GAIN * make_click(sample_rate)).astype(np.float32)
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
