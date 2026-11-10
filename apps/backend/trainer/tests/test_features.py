import numpy as np

from src.core.config import CONFIG
from src.services.training.features import (
    _build_log_filterbank,
    compute_features,
    count_bands,
    make_targets,
)


def test_compute_features_has_one_frame_per_hop() -> None:
    signal = (
        np.random.default_rng(0).uniform(-1, 1, CONFIG.sr).astype(np.float32)
    )

    features = compute_features(signal, CONFIG.features, CONFIG.sr)

    frames = 1 + len(signal) // CONFIG.features.hop
    assert features.shape == (3, count_bands(CONFIG.features), frames)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_every_band_has_a_bin_in_every_window() -> None:
    for n_fft in CONFIG.features.windows:
        filters = _build_log_filterbank(
            n_fft,
            CONFIG.features.fmin,
            count_bands(CONFIG.features),
            CONFIG.features.bands_per_octave,
            CONFIG.sr,
        )

        assert filters.shape == (
            count_bands(CONFIG.features),
            n_fft // 2 + 1,
        )
        assert (filters.sum(axis=1) > 0).all()


def test_make_targets_widens_each_onset_to_its_neighbors() -> None:
    hop_seconds = CONFIG.features.hop / CONFIG.sr

    targets = make_targets(
        np.array([0.0, 10 * hop_seconds]),
        frames=12,
        config=CONFIG.features,
        sr=CONFIG.sr,
    )

    expected = np.zeros(12, dtype=np.float32)
    expected[[0, 10]] = 1.0
    expected[[1, 9, 11]] = CONFIG.features.neighbor_target
    np.testing.assert_array_equal(targets, expected)


def test_make_targets_uses_the_supplied_sample_rate() -> None:
    config = CONFIG.features.model_copy(update={"hop": 100})

    targets = make_targets(np.array([0.5]), frames=10, config=config, sr=1000)

    assert targets[5] == 1.0
    assert targets[4] == targets[6] == config.neighbor_target
