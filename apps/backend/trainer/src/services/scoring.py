from collections.abc import Sequence
from dataclasses import dataclass

import librosa
import mir_eval
import numpy as np
import torch
from scipy.signal import find_peaks

from src.core.config import FeatureConfig, TrainConfig
from src.services.training.network import KickNet

type Probabilities = np.ndarray | Sequence[np.ndarray]


@dataclass(frozen=True)
class Evaluation:
    f_measure: float
    precision: float
    recall: float
    threshold: float


def pick_onsets(
    probabilities: np.ndarray,
    threshold: float,
    config: TrainConfig,
    feature_config: FeatureConfig,
    sr: int,
) -> np.ndarray:
    """Return spaced peak times above the threshold, in seconds."""
    hop = feature_config.hop
    distance = max(1, round(config.min_onset_distance * sr / hop))
    peaks, _ = find_peaks(probabilities, height=threshold, distance=distance)
    return librosa.frames_to_time(peaks, sr=sr, hop_length=hop)


@torch.no_grad()
def predict(
    model: KickNet, features: torch.Tensor, batch_size: int
) -> np.ndarray:
    """Return kick probabilities for each drop and frame."""
    model.eval()
    device = next(model.parameters()).device
    batches = [
        torch.sigmoid(model(batch.to(device))).cpu()
        for batch in features.split(batch_size)
    ]
    return torch.cat(batches).numpy()


def select_labeled_onsets(
    estimated: np.ndarray,
    reference: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Keep estimates within label spans, with tolerance at each end.

    Split spans at label gaps longer than 5 seconds.
    """
    if not len(reference):
        return estimated[:0]
    labels = np.sort(reference)
    groups = np.split(labels, np.flatnonzero(np.diff(labels) > 5.0) + 1)
    keep = np.zeros(len(estimated), dtype=bool)
    for group in groups:
        keep |= (estimated >= group[0] - tolerance) & (
            estimated <= group[-1] + tolerance
        )
    return estimated[keep]


def score_onsets(
    estimated: np.ndarray,
    reference: np.ndarray,
    threshold: float,
    tolerance: float,
) -> Evaluation:
    """Score estimates against reference onsets."""
    selected = select_labeled_onsets(estimated, reference, tolerance)
    f_measure, precision, recall = mir_eval.onset.f_measure(
        reference, selected, window=tolerance
    )
    return Evaluation(f_measure, precision, recall, threshold)


def _evaluate_thresholds(
    probabilities: Probabilities,
    references: Sequence[np.ndarray],
    config: TrainConfig,
    feature_config: FeatureConfig,
    sr: int,
    *,
    labeled_spans_only: bool,
    initial_f_measure: float,
) -> Evaluation:
    best = Evaluation(initial_f_measure, 0.0, 0.0, config.thresholds[0])
    reference_count = sum(len(reference) for reference in references)
    for threshold in config.thresholds:
        detected = hits = 0
        for probability, reference in zip(
            probabilities, references, strict=True
        ):
            estimated = pick_onsets(
                probability, threshold, config, feature_config, sr
            )
            if labeled_spans_only:
                estimated = select_labeled_onsets(
                    estimated, reference, config.tolerance
                )
            detected += len(estimated)
            hits += len(
                mir_eval.util.match_events(
                    reference, estimated, config.tolerance
                )
            )
        precision = hits / max(detected, 1)
        recall = hits / max(reference_count, 1)
        f_measure = 2 * precision * recall / max(precision + recall, 1e-9)
        if f_measure > best.f_measure:
            best = Evaluation(f_measure, precision, recall, threshold)
    return best


def evaluate(
    probabilities: Probabilities,
    references: Sequence[np.ndarray],
    config: TrainConfig,
    feature_config: FeatureConfig,
    sr: int,
) -> Evaluation:
    """Choose the best threshold over all reference onsets."""
    return _evaluate_thresholds(
        probabilities,
        references,
        config,
        feature_config,
        sr,
        labeled_spans_only=False,
        initial_f_measure=0.0,
    )


def evaluate_labeled(
    probabilities: Probabilities,
    references: Sequence[np.ndarray],
    config: TrainConfig,
    feature_config: FeatureConfig,
    sr: int,
) -> Evaluation:
    """Choose the best threshold inside labeled spans."""
    return _evaluate_thresholds(
        probabilities,
        references,
        config,
        feature_config,
        sr,
        labeled_spans_only=True,
        initial_f_measure=-1.0,
    )
