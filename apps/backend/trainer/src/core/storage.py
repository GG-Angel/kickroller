from pathlib import Path

import librosa
import numpy as np
import yaml
from loguru import logger
from scipy.io import wavfile

from src.core.config import CFG
from src.models.audio import Signal


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_audio(path: Path, sr: int = CFG.sr) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sr)
    logger.info(f"Loaded audio from {path}")
    return signal


def save_audio(path: Path, signal: Signal, sr: int = CFG.sr) -> None:
    wavfile.write(path, sr, signal)
    logger.info(f"Saved audio to {path}")
