from pathlib import Path

import librosa
import numpy as np
import yaml
from loguru import logger
from scipy.io import wavfile

from src.models.audio import Signal


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_audio(path: Path, sr: int) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sr)
    logger.debug(f"Loaded audio from {path}")
    return signal


def save_audio(path: Path, signal: Signal, sr: int) -> None:
    wavfile.write(path, sr, signal)
    logger.info(f"Saved audio to {path}")


def save_onsets(
    path: Path,
    onsets: np.ndarray,
    confidences: np.ndarray | None = None,
) -> None:
    rows = (
        onsets
        if confidences is None
        else np.column_stack([onsets, confidences])
    )
    np.savetxt(path, rows, fmt="%.6f", delimiter=",")
    logger.info(f"Saved {len(onsets)} onsets to {path}")


def load_onsets(path: Path) -> np.ndarray:
    """Load onsets from a CSV file, including Sonic Visualizer exports."""
    return np.loadtxt(path, delimiter=",", usecols=0, ndmin=1)
