from src.models.audio import Signal
import librosa
from pathlib import Path

import numpy as np

from src.core.config import CFG
from scipy.io import wavfile


def load_audio(path: Path, sr: int = CFG.sr) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sr)
    return signal


def save_audio(path: Path, signal: Signal, sr: int = CFG.sr) -> None:
    wavfile.write(path, sr, signal)
