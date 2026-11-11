from functools import cache

import librosa
import numpy as np

from src.core.config import FeatureConfig
from src.models.audio import Signal


def count_bands(config: FeatureConfig) -> int:
    octaves = np.log2(config.fmax / config.fmin)
    return int(octaves * config.bands_per_octave) + 1


@cache
def _build_log_filterbank(
    n_fft: int, fmin: float, bands: int, bands_per_octave: int, sr: int
) -> np.ndarray:
    """Build log-spaced triangular filters with shape (bands, FFT bins)."""
    # Add edge frequencies for the first and last bands.
    frequencies = librosa.cqt_frequencies(
        bands + 2,
        fmin=fmin * 2 ** (-1 / bands_per_octave),
        bins_per_octave=bands_per_octave,
    )
    low, center, high = (
        frequencies[:-2, None],
        frequencies[1:-1, None],
        frequencies[2:, None],
    )
    bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    weights = np.maximum(
        0.0,
        np.minimum(
            (bins - low) / (center - low), (high - bins) / (high - center)
        ),
    )
    for band in np.flatnonzero(weights.sum(axis=1) == 0):
        weights[band, np.argmin(np.abs(bins - center[band, 0]))] = 1.0
    return weights.astype(np.float32)


def compute_features(
    signal: Signal, config: FeatureConfig, sr: int
) -> np.ndarray:
    """Build log spectrograms with shape (windows, bands, frames)."""
    layers = []
    for n_fft in config.windows:
        magnitude = np.abs(
            librosa.stft(signal, n_fft=n_fft, hop_length=config.hop)
        )
        filters = _build_log_filterbank(
            n_fft,
            config.fmin,
            count_bands(config),
            config.bands_per_octave,
            sr,
        )
        bands = filters @ magnitude
        layers.append(librosa.amplitude_to_db(bands, amin=1e-5, top_db=None))
    return np.stack(layers).astype(np.float32)


def make_targets(
    onsets: np.ndarray,
    frames: int,
    config: FeatureConfig,
    sr: int,
) -> np.ndarray:
    """Mark each onset frame and its neighbors."""
    targets = np.zeros(frames, dtype=np.float32)
    at = np.rint(onsets * sr / config.hop).astype(int)
    for offset, value in (
        (-1, config.neighbor_target),
        (1, config.neighbor_target),
        (0, 1.0),
    ):
        near = at + offset
        near = near[(near >= 0) & (near < frames)]
        targets[near] = np.maximum(targets[near], value)
    return targets
