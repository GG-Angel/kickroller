from dataclasses import dataclass

import librosa
import mir_eval
import numpy as np
import torch

from src.core.config import CONFIG
from src.models.audio import Signal
from src.services.training.evaluate import Evaluation, pick_onsets, predict
from src.services.training.features import compute_features
from src.services.training.network import Checkpoint


@dataclass(frozen=True)
class Analysis:
    onsets: np.ndarray
    """Kick onsets in seconds."""
    confidences: np.ndarray
    """Kick probability at each onset, from 0 to 1."""


def analyze_signal(
    signal: Signal, checkpoint: Checkpoint, threshold: float | None = None
) -> Analysis:
    """Find the kick onsets of a track."""
    features = torch.from_numpy(compute_features(signal))
    probabilities = predict(checkpoint.model, features[None], batch_size=1)[0]
    if threshold is None:
        threshold = checkpoint.threshold
    onsets = pick_onsets(probabilities, threshold)
    frames = librosa.time_to_frames(
        onsets,
        sr=CONFIG.sr,
        hop_length=CONFIG.features.hop,
    )
    return Analysis(onsets=onsets, confidences=probabilities[frames])


def select_labeled_onsets(
    estimated: np.ndarray,
    reference: np.ndarray,
    tolerance: float = CONFIG.train.tolerance,
) -> np.ndarray:
    """Keep predictions in label groups separated by gaps longer than 5 s.

    Each span runs from its first to last label, plus ``tolerance`` at
    each end. An empty reference has no scored spans.
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
    tolerance: float = CONFIG.train.tolerance,
) -> Evaluation:
    """Score only labeled spans, with a hit within ``tolerance``."""
    estimated = select_labeled_onsets(estimated, reference, tolerance)
    f_measure, precision, recall = mir_eval.onset.f_measure(
        reference, estimated, window=tolerance
    )
    return Evaluation(f_measure, precision, recall, threshold)


def add_clicks(signal: Signal, onsets: np.ndarray) -> Signal:
    """The signal with a click at each onset, for listening.

    The signal is 12 dB quieter, so that the clicks are audible over a loud
    master, and the peak stays at or below 1.
    """
    clicks = librosa.clicks(times=onsets, sr=CONFIG.sr, length=len(signal))
    quiet = librosa.util.normalize(signal) * 0.25
    return (quiet + (clicks * 0.75)).astype(np.float32)
