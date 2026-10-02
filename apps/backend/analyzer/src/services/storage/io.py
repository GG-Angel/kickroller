from pathlib import Path

import librosa

from core.settings import SETTINGS


def load_file(path: str | Path, sample_rate: int = SETTINGS.sample_rate):
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal
