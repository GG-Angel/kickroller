import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from src.core.config import CONFIG
from src.models.audio import Signal


def stretch(signal: Signal, from_bpm: int, to_bpm: int) -> Signal:
    rate = to_bpm / from_bpm
    stretched = librosa.effects.time_stretch(y=signal, rate=rate)
    return stretched


def trim(signal: Signal) -> Signal:
    trimmed, _ = librosa.effects.trim(signal)
    return trimmed


def filter(
    signal: Signal,
    cutoff: float,
    btype: str,
    order: int = 2,
    sr: int = CONFIG.sr,
) -> Signal:
    sos = butter(order, cutoff, btype, fs=sr, output="sos")
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


def normalize(signal: Signal) -> Signal:
    return librosa.util.normalize(signal)
