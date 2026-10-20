from functools import cache

import librosa
import numpy as np

from src.core.config import CONFIG, FeatureConfig
from src.models.audio import Signal


def band_count(config: FeatureConfig = CONFIG.features) -> int:
    octaves = np.log2(config.fmax / config.fmin)
    return int(octaves * config.bands_per_octave) + 1


@cache
def _log_filterbank(
    n_fft: int, fmin: float, bands: int, bands_per_octave: int, sr: int
) -> np.ndarray:
    """Triangular bands on a log scale, of shape (bands, FFT bins).

    A band narrower than one FFT bin uses its nearest bin, so that each window
    length has the same bands.
    """
    # one extra frequency on each side gives the edges of the outer bands
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
    signal: Signal, config: FeatureConfig = CONFIG.features
) -> np.ndarray:
    """Log spectrograms of shape (windows, bands, frames).

    Frame ``t`` is centered at sample ``t * hop``, so a drop has
    ``1 + len(signal) // hop`` frames.
    """
    layers = []
    for n_fft in config.windows:
        magnitude = np.abs(
            librosa.stft(signal, n_fft=n_fft, hop_length=config.hop)
        )
        filters = _log_filterbank(
            n_fft,
            config.fmin,
            band_count(config),
            config.bands_per_octave,
            CONFIG.sr,
        )
        bands = filters @ magnitude
        layers.append(librosa.amplitude_to_db(bands, amin=1e-5, top_db=None))
    return np.stack(layers).astype(np.float32)


def make_targets(
    onsets: np.ndarray,
    frames: int,
    config: FeatureConfig = CONFIG.features,
    sr: int = CONFIG.sr,
) -> np.ndarray:
    """1 at the frame of each onset and ``neighbor_target`` at its neighbors."""
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
