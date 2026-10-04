from pathlib import Path

import librosa
import numpy as np

from core.settings import SETTINGS


def load_audio_file(path: Path, sample_rate: int = SETTINGS.sample_rate) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal
