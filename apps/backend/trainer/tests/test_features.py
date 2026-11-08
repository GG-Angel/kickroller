import numpy as np

from src.core.config import CONFIG
from src.services.training.features import (
    _log_filterbank,
    band_count,
    compute_features,
    make_targets,
)


def test_compute_features_has_one_frame_per_hop() -> None:
    signal = (
        np.random.default_rng(0).uniform(-1, 1, CONFIG.sr).astype(np.float32)
    )

    features = compute_features(signal)

    frames = 1 + len(signal) // CONFIG.features.hop
    assert features.shape == (3, band_count(), frames)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_every_band_has_a_bin_in_every_window() -> None:
    for n_fft in CONFIG.features.windows:
        filters = _log_filterbank(
            n_fft,
            CONFIG.features.fmin,
            band_count(),
            CONFIG.features.bands_per_octave,
            CONFIG.sr,
        )

        assert filters.shape == (band_count(), n_fft // 2 + 1)
        assert (filters.sum(axis=1) > 0).all()


def test_make_targets_widens_each_onset_to_its_neighbors() -> None:
    hop_seconds = CONFIG.features.hop / CONFIG.sr

    targets = make_targets(np.array([0.0, 10 * hop_seconds]), frames=12)

    expected = np.zeros(12, dtype=np.float32)
    expected[[0, 10]] = 1.0
    expected[[1, 9, 11]] = CONFIG.features.neighbor_target
    np.testing.assert_array_equal(targets, expected)
