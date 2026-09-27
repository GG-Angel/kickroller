import numpy as np

from analyzer.odf import bin_range, stft_bins


def low_band_profile(
    signal: np.ndarray, sample_rate: int, n_fft: int, hop: int, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Energy and spectral centroid (Hz) of the low band, one value per hop."""
    first, last = bin_range(n_fft, sample_rate, fmin, fmax)
    magnitude = stft_bins(signal, n_fft, hop, first, last)
    freqs = np.arange(first, last + 1) * sample_rate / n_fft
    energy = (magnitude.astype(np.float64) ** 2).sum(axis=1)
    centroid = (magnitude * freqs).sum(axis=1) / (magnitude.sum(axis=1) + 1e-9)
    return energy, centroid


def low_band_change(
    frames: np.ndarray, energy: np.ndarray, centroid: np.ndarray, context_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-band change across each attack: `context_frames` after versus before.

    Returns (level_db, rise_db, pitch_jump_hz) per frame:
    level_db is the energy after, relative to the track's 95th percentile;
    rise_db is the energy after relative to before;
    pitch_jump_hz is the centroid after minus before.
    """
    if len(frames) == 0 or len(energy) == 0:
        empty = np.empty(0)
        return empty, empty, empty
    reference = max(float(np.percentile(energy, 95)), 1e-30)
    post = np.minimum(frames + context_frames, len(energy) - 1)
    pre = np.maximum(frames - context_frames, 0)
    eps = 1e-12 * reference
    level_db = 10.0 * np.log10(energy[post] / reference + 1e-12)
    rise_db = 10.0 * np.log10((energy[post] + eps) / (energy[pre] + eps))
    return level_db, rise_db, centroid[post] - centroid[pre]


def low_band_gap_rise(
    frames: np.ndarray,
    energy: np.ndarray,
    frames_after: int = 3,
    frames_before: int = 2,
) -> np.ndarray:
    """Short-gap rise in dB at each attack.

    Compares the highest energy 1 to `frames_after` frames after the attack with
    the lowest energy from the attack frame back to `frames_before` frames before.
    This finds a low band that starts with the click (for example a regular drum
    kick), where the 30 ms rise of `low_band_change` can miss it.
    """
    if len(frames) == 0 or len(energy) == 0:
        return np.empty(0)
    last = len(energy) - 1
    post = np.max(
        [energy[np.minimum(frames + k, last)] for k in range(1, frames_after + 1)],
        axis=0,
    )
    pre = np.min(
        [energy[np.maximum(frames - k, 0)] for k in range(frames_before + 1)], axis=0
    )
    eps = 1e-12 * max(float(np.percentile(energy, 95)), 1e-30)
    return 10.0 * np.log10((post + eps) / (pre + eps))
