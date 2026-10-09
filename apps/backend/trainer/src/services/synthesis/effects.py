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


def cut(signal: Signal, length: int) -> Signal:
    if len(signal) <= length:
        return signal  # signal is already shorter than desired length

    fade = min(round(0.01 * CONFIG.sr), length)
    cut_signal = signal[:length].copy()
    cut_signal[length - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return cut_signal


def filter(
    signal: Signal,
    cutoff: float,
    btype: str,
    order: int = 2,
) -> Signal:
    sos = butter(order, cutoff, btype, fs=CONFIG.sr, output="sos")
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


def normalize(signal: Signal) -> Signal:
    return librosa.util.normalize(signal)
