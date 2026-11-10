import numpy as np

from src.core.config import CONFIG
from src.services.scoring import evaluate, pick_onsets

FPS = CONFIG.sr / CONFIG.features.hop


def test_pick_onsets_keeps_peaks_above_the_threshold() -> None:
    probabilities = np.zeros(100)
    probabilities[[10, 30]] = [0.9, 0.2]

    onsets = pick_onsets(
        probabilities, 0.5, CONFIG.train, CONFIG.features, CONFIG.sr
    )

    np.testing.assert_allclose(onsets, [10 / FPS])


def test_pick_onsets_keeps_the_highest_of_close_peaks() -> None:
    probabilities = np.zeros(100)
    probabilities[[10, 12, 20]] = [0.6, 0.9, 0.8]

    onsets = pick_onsets(
        probabilities, 0.5, CONFIG.train, CONFIG.features, CONFIG.sr
    )

    np.testing.assert_allclose(onsets, [12 / FPS, 20 / FPS])


def test_pick_onsets_uses_the_supplied_hop_and_sample_rate() -> None:
    probabilities = np.zeros(10)
    probabilities[2] = 0.9
    feature_config = CONFIG.features.model_copy(update={"hop": 500})

    onsets = pick_onsets(
        probabilities, 0.5, CONFIG.train, feature_config, sr=1000
    )

    np.testing.assert_allclose(onsets, [1.0])


def test_evaluate_scores_perfect_probabilities_as_one() -> None:
    probabilities = np.zeros((2, 200))
    probabilities[0, [20, 60]] = 1.0
    probabilities[1, [100]] = 1.0
    onsets = [np.array([20, 60]) / FPS, np.array([100]) / FPS]

    evaluation = evaluate(
        probabilities, onsets, CONFIG.train, CONFIG.features, CONFIG.sr
    )

    assert evaluation.f_measure == 1.0
    assert evaluation.precision == evaluation.recall == 1.0


def test_evaluate_counts_misses_and_false_onsets() -> None:
    probabilities = np.zeros((1, 200))
    probabilities[0, [20, 150]] = 1.0
    onsets = [np.array([20, 60]) / FPS]

    evaluation = evaluate(
        probabilities, onsets, CONFIG.train, CONFIG.features, CONFIG.sr
    )

    assert evaluation.precision == 0.5
    assert evaluation.recall == 0.5
