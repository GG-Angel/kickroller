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
    """Find the kick onsets of a track, with the threshold of the checkpoint
    if ``threshold`` is None."""
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


def score_onsets(
    estimated: np.ndarray,
    reference: np.ndarray,
    threshold: float,
    tolerance: float = CONFIG.train.tolerance,
) -> Evaluation:
    """Score the onsets against labels, with a hit within ``tolerance``."""
    # shortcut: scores one span from the first to the last label, not the
    # labeled drop regions; read the region files when they exist
    if len(reference):
        start, end = reference.min() - tolerance, reference.max() + tolerance
        estimated = estimated[(estimated >= start) & (estimated <= end)]
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
    return (quiet + 0.75 * clicks).astype(np.float32)
