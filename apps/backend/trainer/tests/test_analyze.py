from pathlib import Path

import numpy as np
import pytest
import torch

from src.core.config import CONFIG
from src.core.storage import load_onsets, save_onsets
from src.services.analysis.analyze import (
    add_clicks,
    analyze_signal,
)
from src.services.scoring import score_onsets, select_labeled_onsets
from src.services.training.network import Checkpoint, KickNet

FPS = CONFIG.sr / CONFIG.features.hop


class _FixedNet(KickNet):
    """Logits with peaks at frames 20 (high) and 60 (low)."""

    def __init__(self) -> None:
        super().__init__(CONFIG.features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = torch.full(features.shape[:1] + features.shape[-1:], -10.0)
        logits[:, 20] = 3.0
        logits[:, 60] = 0.0
        return logits


def _checkpoint(threshold: float) -> Checkpoint:
    return Checkpoint(model=_FixedNet().eval(), threshold=threshold)


def test_analyze_signal_uses_the_checkpoint_threshold() -> None:
    signal = np.zeros(CONFIG.sr, dtype=np.float32)

    analysis = analyze_signal(
        signal, _checkpoint(0.8), None, CONFIG.train, CONFIG.sr
    )

    np.testing.assert_allclose(analysis.onsets, [20 / FPS])
    np.testing.assert_allclose(
        analysis.confidences, [torch.sigmoid(torch.tensor(3.0))]
    )


def test_analyze_signal_can_override_the_threshold() -> None:
    signal = np.zeros(CONFIG.sr, dtype=np.float32)

    analysis = analyze_signal(
        signal, _checkpoint(0.8), 0.4, CONFIG.train, CONFIG.sr
    )

    np.testing.assert_allclose(analysis.onsets, [20 / FPS, 60 / FPS])
    np.testing.assert_allclose(analysis.confidences[1], 0.5)


def test_score_onsets_ignores_onsets_outside_the_labels() -> None:
    reference = np.array([10.0, 10.375, 10.75])
    estimated = np.array([2.0, 10.005, 10.375, 10.75, 30.0])

    score = score_onsets(estimated, reference, 0.5, CONFIG.train.tolerance)

    assert score.f_measure == pytest.approx(1.0)
    assert score.threshold == 0.5


def test_score_onsets_counts_misses_and_false_onsets() -> None:
    reference = np.array([1.0, 2.0])
    estimated = np.array([1.0, 1.5])

    score = score_onsets(estimated, reference, 0.5, CONFIG.train.tolerance)

    assert score.precision == score.recall == 0.5


def test_score_onsets_ignores_unlabeled_gaps_between_drops() -> None:
    reference = np.array([10.0, 10.375, 30.0, 30.375])
    estimated = np.array([10.0, 10.375, 15.0, 20.0, 30.0, 30.375])

    score = score_onsets(estimated, reference, 0.5, CONFIG.train.tolerance)

    assert score.f_measure == 1.0


def test_score_onsets_keeps_pauses_of_five_seconds() -> None:
    reference = np.array([10.0, 15.0])
    estimated = np.array([10.0, 12.0, 15.0])

    score = score_onsets(estimated, reference, 0.5, CONFIG.train.tolerance)

    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == 1.0


def test_select_labeled_onsets_includes_span_edges() -> None:
    reference = np.array([10.0, 11.0, 30.0])
    estimated = np.array(
        [9.979, 9.98, 11.02, 11.021, 29.979, 29.98, 30.02, 30.021]
    )

    selected = select_labeled_onsets(
        estimated, reference, CONFIG.train.tolerance
    )

    np.testing.assert_allclose(selected, [9.98, 11.02, 29.98, 30.02])


def test_select_labeled_onsets_has_no_spans_without_labels() -> None:
    selected = select_labeled_onsets(
        np.array([1.0]), np.array([]), CONFIG.train.tolerance
    )

    assert selected.size == 0


def test_add_clicks_keeps_the_length() -> None:
    signal = np.zeros(CONFIG.sr, dtype=np.float32)

    clicked = add_clicks(signal, np.array([0.5]), CONFIG.sr)

    assert len(clicked) == len(signal)
    assert clicked.dtype == np.float32
    assert np.abs(clicked[CONFIG.sr // 2 : CONFIG.sr // 2 + 100]).max() > 0
    assert np.abs(clicked[: CONFIG.sr // 4]).max() == 0


def test_add_clicks_are_louder_than_a_loud_track() -> None:
    signal = np.random.default_rng(0).uniform(-1, 1, CONFIG.sr)

    clicked = add_clicks(signal.astype(np.float32), np.array([0.5]), CONFIG.sr)

    start = CONFIG.sr // 2
    click = clicked[start : start + 200] - 0.25 * signal[start : start + 200]
    assert np.abs(clicked).max() <= 1.0
    assert np.abs(click).max() > 2 * 0.25


def test_onsets_round_trip_with_confidences(tmp_path: Path) -> None:
    path = tmp_path / "onsets.csv"

    save_onsets(path, np.array([0.5, 1.25]), np.array([0.9, 0.6]))

    np.testing.assert_allclose(load_onsets(path), [0.5, 1.25])
    assert path.read_text().splitlines()[0] == "0.500000,0.900000"


def test_load_onsets_reads_sonic_visualiser_exports(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("0.500000,1,kick\n")

    np.testing.assert_allclose(load_onsets(path), [0.5])
