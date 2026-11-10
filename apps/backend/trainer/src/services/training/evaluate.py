from dataclasses import dataclass

import librosa
import mir_eval
import numpy as np
import torch
from scipy.signal import find_peaks

from src.core.config import CONFIG, TrainConfig
from src.services.training.network import KickNet


@dataclass(frozen=True)
class Evaluation:
    f_measure: float
    precision: float
    recall: float
    threshold: float


def pick_onsets(
    probabilities: np.ndarray,
    threshold: float,
    config: TrainConfig = CONFIG.train,
    sr: int = CONFIG.sr,
) -> np.ndarray:
    """Onset times in seconds: peaks above ``threshold``, spaced apart."""
    hop = CONFIG.features.hop
    distance = max(1, round(config.min_onset_distance * sr / hop))
    peaks, _ = find_peaks(probabilities, height=threshold, distance=distance)
    return librosa.frames_to_time(peaks, sr=sr, hop_length=hop)


@torch.no_grad()
def predict(
    model: KickNet, features: torch.Tensor, batch_size: int
) -> np.ndarray:
    """Kick probabilities (drops, frames) of features (drops, ...)."""
    model.eval()
    device = next(model.parameters()).device
    batches = [
        torch.sigmoid(model(batch.to(device))).cpu()
        for batch in features.split(batch_size)
    ]
    return torch.cat(batches).numpy()


def evaluate(
    probabilities: np.ndarray,
    onsets: list[np.ndarray],
    config: TrainConfig = CONFIG.train,
) -> Evaluation:
    """The scores of the threshold with the best F-measure over all drops."""
    best = Evaluation(0.0, 0.0, 0.0, config.thresholds[0])
    references = sum(len(reference) for reference in onsets)
    for threshold in config.thresholds:
        detected = hits = 0
        for probability, reference in zip(probabilities, onsets, strict=True):
            estimated = pick_onsets(probability, threshold, config)
            detected += len(estimated)
            hits += len(
                mir_eval.util.match_events(
                    reference, estimated, config.tolerance
                )
            )
        precision = hits / max(detected, 1)
        recall = hits / max(references, 1)
        f_measure = 2 * precision * recall / max(precision + recall, 1e-9)
        if f_measure > best.f_measure:
            best = Evaluation(f_measure, precision, recall, threshold)
    return best
