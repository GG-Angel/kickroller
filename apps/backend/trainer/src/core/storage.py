from pathlib import Path

import librosa
import numpy as np
import yaml
from loguru import logger
from scipy.io import wavfile

from src.core.config import CONFIG
from src.models.audio import Signal


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_audio(path: Path) -> np.ndarray:
    signal, _ = librosa.load(path, sr=CONFIG.sr)
    logger.info(f"Loaded audio from {path}")
    return signal


def save_audio(path: Path, signal: Signal) -> None:
    wavfile.write(path, CONFIG.sr, signal)
    logger.info(f"Saved audio to {path}")
