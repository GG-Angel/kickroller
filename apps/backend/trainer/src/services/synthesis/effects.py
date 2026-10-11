from dataclasses import dataclass, field
from typing import Protocol

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from src.core.config import CONFIG
from src.models.audio import Signal


def cut(
    signal: Signal,
    length: int,
    fade: float = 0.01,
    sr: int = CONFIG.sr,
) -> Signal:
    if len(signal) <= length:
        return signal  # signal is already shorter than desired length

    fade_samples = min(round(fade * sr), length)
    cut_signal = signal[:length].copy()
    cut_signal[length - fade_samples :] *= np.linspace(
        1.0, 0.0, fade_samples, dtype=np.float32
    )
    return cut_signal


def trim(signal: Signal) -> Signal:
    trimmed, _ = librosa.effects.trim(signal)
    return trimmed


def filter(
    signal: Signal,
    cutoff: float | tuple[float, float],
    btype: str,
    order: int = 2,
    sr: int = CONFIG.sr,
) -> Signal:
    sos = butter(order, cutoff, btype, fs=sr, output="sos")
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


def normalize(signal: Signal) -> Signal:
    return librosa.util.normalize(signal)


def sidechain(
    signal: Signal,
    triggers: np.ndarray,
    ratio: float = 0.5,
    duration: float = 0.1,
    fade: float = 0.005,
    sr: int = CONFIG.sr,
) -> Signal:
    _validate_sidechain(triggers, ratio, duration)

    # only sidechain if both ratio and duration are non-zero
    if ratio == 0 or duration == 0:
        return signal

    samples = len(signal)
    duration_samples = max(1, round(duration * sr))
    fade_samples = min(max(1, round(fade * sr)), duration_samples // 2)
    envelope = np.zeros(samples, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(
        np.pi * np.linspace(0.0, 1.0, fade_samples + 1, dtype=np.float32)
    )
    for trigger in triggers:
        start = int(trigger)
        end = min(start + duration_samples, samples)
        length = end - start
        if length <= 0:
            continue
        event_envelope = np.ones(length, dtype=np.float32)
        if fade_samples:
            event_fade = min(fade_samples, length // 2)
            if event_fade:
                event_envelope[:event_fade] = ramp[:event_fade]
                event_envelope[-event_fade:] = ramp[:event_fade][::-1]
        envelope[start:end] = np.maximum(envelope[start:end], event_envelope)
    return np.asarray(signal * (1.0 - ratio * envelope), dtype=np.float32)


def _validate_sidechain(
    triggers: np.ndarray, ratio: float, duration: float
) -> None:
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("Sidechain ratio must be between 0 and 1.")
    if not np.isfinite(duration) or duration < 0:
        raise ValueError(
            "Sidechain duration must be a finite non-negative value."
        )
    if np.any(np.asarray(triggers) < 0):
        raise ValueError(
            "Sidechain triggers must be non-negative sample indices."
        )


class Effect(Protocol):
    def process(self, signal: Signal, sr: int) -> Signal: ...


@dataclass(frozen=True)
class Cut(Effect):
    length: int
    fade: float = 0.01

    def process(self, signal: Signal, sr: int) -> Signal:
        return cut(signal, self.length, self.fade, sr=sr)


@dataclass(frozen=True)
class Trim(Effect):
    def process(self, signal: Signal, sr: int) -> Signal:
        return trim(signal)


@dataclass(frozen=True)
class Filter(Effect):
    cutoff: float | tuple[float, float]
    btype: str
    order: int = 2

    def process(self, signal: Signal, sr: int) -> Signal:
        return filter(signal, self.cutoff, self.btype, self.order, sr=sr)


@dataclass(frozen=True)
class Normalize(Effect):
    def process(self, signal: Signal, sr: int) -> Signal:
        return normalize(signal)


@dataclass(frozen=True, eq=False)
class Sidechain(Effect):
    triggers: np.ndarray = field(repr=False)
    ratio: float = 0.5
    duration: float = 0.1
    fade: float = 0.005

    def __post_init__(self) -> None:
        _validate_sidechain(self.triggers, self.ratio, self.duration)

    def process(self, signal: Signal, sr: int) -> Signal:
        return sidechain(
            signal,
            self.triggers,
            self.ratio,
            self.duration,
            self.fade,
            sr=sr,
        )
