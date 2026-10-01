"""Peak picking on a frame activation."""

import numpy as np
from scipy.ndimage import maximum_filter1d


def local_peaks(odf: np.ndarray, threshold: float, local_max_frames: int) -> np.ndarray:
    """Frame indices that are the maximum within +/- `local_max_frames` and at least `threshold`."""
    if len(odf) == 0:
        return np.empty(0, dtype=int)
    local_max = maximum_filter1d(odf, size=2 * local_max_frames + 1, mode="nearest")
    return np.flatnonzero((odf >= local_max) & (odf >= threshold))


def interpolate_peaks(odf: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Peak positions between frames, in fractional frames.

    Quadratic interpolation: a parabola through the log activation of each peak
    frame and its two neighbors (J. O. Smith, "Quadratic Interpolation of
    Spectral Peaks"). A peak at the first or last frame stays on its frame.
    """
    positions = frames.astype(float)
    inner = (frames > 0) & (frames < len(odf) - 1)
    log = np.log(np.maximum(odf, 1e-9))
    left, center, right = (log[frames[inner] + i] for i in (-1, 0, 1))
    curvature = left - 2.0 * center + right
    safe = np.where(curvature < 0, curvature, -1.0)
    shift = np.where(curvature < 0, 0.5 * (left - right) / safe, 0.0)
    positions[inner] += np.clip(shift, -0.5, 0.5)
    return positions


def enforce_min_distance(
    frames: np.ndarray, strength: np.ndarray, min_distance_frames: int
) -> np.ndarray:
    """Remove peaks closer than `min_distance_frames` to a stronger peak.

    `strength` has one value per frame in `frames`. Returns the indices of the
    kept peaks into `frames`, sorted by frame.
    """
    kept: list[int] = []
    for i in np.argsort(-strength, kind="stable"):
        if all(abs(frames[i] - frames[k]) >= min_distance_frames for k in kept):
            kept.append(int(i))
    kept_array = np.asarray(kept, dtype=int)
    return kept_array[np.argsort(frames[kept_array], kind="stable")]
