import numpy as np
from scipy.ndimage import maximum_filter1d


def local_peaks(odf: np.ndarray, threshold: float, local_max_frames: int) -> np.ndarray:
    """Frame indices that are the maximum within +/- `local_max_frames` and at least `threshold`."""
    if len(odf) == 0:
        return np.empty(0, dtype=int)
    local_max = maximum_filter1d(odf, size=2 * local_max_frames + 1, mode="nearest")
    return np.flatnonzero((odf >= local_max) & (odf >= threshold))


def enforce_min_distance(
    frames: np.ndarray, strength: np.ndarray, min_distance_frames: int
) -> np.ndarray:
    """Remove peaks closer than `min_distance_frames` to a stronger peak.

    `strength` has one value per frame in `frames`. Returns sorted frames.
    """
    kept: list[int] = []
    for i in np.argsort(-strength, kind="stable"):
        if all(abs(frames[i] - k) >= min_distance_frames for k in kept):
            kept.append(int(frames[i]))
    return np.sort(np.asarray(kept, dtype=int))
