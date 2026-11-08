from dataclasses import dataclass

import librosa
import numpy as np

from src.core.config import CONFIG
from src.models.audio import LabeledDrop


@dataclass(frozen=True)
class CropWindow:
    start: float
    """Start of the window in seconds."""
    end: float
    """End of the window in seconds."""


def crop_drop(
    drop: LabeledDrop, window: CropWindow, sr: int = CONFIG.sr
) -> LabeledDrop:
    """Cut the window from the drop and shift its onsets.

    The times are rounded to samples, so the onsets stay on the same samples.

    Raises:
        ValueError: If the window is empty or not inside the drop.
    """
    first, last = round(window.start * sr), round(window.end * sr)
    if not 0 <= first < last <= len(drop.signal):
        raise ValueError("The crop window must be inside the drop.")
    start_time, end_time = librosa.samples_to_time([first, last], sr=sr)
    keep = (drop.onsets >= start_time) & (drop.onsets < end_time)
    return LabeledDrop(
        signal=drop.signal[first:last].copy(),
        onsets=drop.onsets[keep] - start_time,
    )


def draw_crop(
    rng: np.random.Generator, duration: float, seconds: float
) -> CropWindow:
    """Return a random window of ``seconds`` inside ``duration`` seconds.

    Raises:
        ValueError: If ``duration`` is shorter than ``seconds``.
    """
    if duration < seconds:
        raise ValueError("The drop is shorter than the crop window.")
    start = rng.uniform(0, duration - seconds)
    return CropWindow(start=start, end=start + seconds)
