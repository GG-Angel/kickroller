from dataclasses import dataclass

import librosa
import numpy as np
import torch
from src.core.config import TrainConfig
from src.models.audio import Signal
from src.services.scoring import pick_onsets, predict
from src.services.training.features import compute_features
from src.services.training.network import Checkpoint


@dataclass(frozen=True)
class Analysis:
    onsets: np.ndarray
    """Onset times in seconds."""
    confidences: np.ndarray
    """Probability at each onset, from 0 to 1."""


def analyze_signal(
    signal: Signal,
    checkpoint: Checkpoint,
    threshold: float | None,
    config: TrainConfig,
    sr: int,
) -> Analysis:
    """Find kick onsets in a track."""
    feature_config = checkpoint.model.config
    features = torch.from_numpy(compute_features(signal, feature_config, sr))
    probabilities = predict(checkpoint.model, features[None], batch_size=1)[0]
    if threshold is None:
        threshold = checkpoint.threshold
    onsets = pick_onsets(probabilities, threshold, config, feature_config, sr)
    frames = librosa.time_to_frames(
        onsets,
        sr=sr,
        hop_length=feature_config.hop,
    )
    return Analysis(onsets=onsets, confidences=probabilities[frames])


def add_clicks(signal: Signal, onsets: np.ndarray, sr: int) -> Signal:
    """Add a click at each onset for listening.

    The clicks are 12 dB quieter and the peak stays at or below 1.
    """
    clicks = librosa.clicks(times=onsets, sr=sr, length=len(signal))
    quiet = librosa.util.normalize(signal) * 0.25
    return (quiet + (clicks * 0.75)).astype(np.float32)
