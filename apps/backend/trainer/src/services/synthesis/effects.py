from dataclasses import dataclass, field
from typing import Protocol

import librosa
import numpy as np
from scipy.signal import bilinear, butter, sosfilt, tf2sos

from src.core.config import CONFIG
from src.models.audio import Signal


def cut_signal(
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


def trim_signal(signal: Signal) -> Signal:
    trimmed, _ = librosa.effects.trim(signal)
    return trimmed


def filter_signal(
    signal: Signal,
    cutoff: float | tuple[float, float],
    btype: str,
    order: int = 2,
    q: float | None = None,
    sr: int = CONFIG.sr,
) -> Signal:
    """Filter with a Butterworth filter, or with a resonant biquad if ``q`` is set.

    The resonant biquad is a 2nd-order low-pass or high-pass, so it ignores
    ``order``. Its gain at the cutoff is ``q``, and ``q = 1/√2`` gives no
    resonance (the same as a 2nd-order Butterworth).
    """
    if q is None:
        sos = butter(order, cutoff, btype, fs=sr, output="sos")
    else:
        sos = _resonant_sos(cutoff, btype, q, sr)
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


def _resonant_sos(
    cutoff: float | tuple[float, float], btype: str, q: float, sr: int
) -> np.ndarray:
    if not isinstance(cutoff, (int, float)) or not 0 < cutoff < sr / 2:
        raise ValueError("A resonant filter needs one cutoff below Nyquist.")
    if btype not in ("lowpass", "highpass"):
        raise ValueError("A resonant filter must be a lowpass or highpass.")
    if not q > 0:
        raise ValueError("Q must be positive.")
    # analog prototype with the cutoff pre-warped for the bilinear transform
    w0 = 2 * sr * np.tan(np.pi * cutoff / sr)
    numerator = [0.0, 0.0, w0**2] if btype == "lowpass" else [1.0, 0.0, 0.0]
    b, a = bilinear(numerator, [1.0, w0 / q, w0**2], fs=sr)
    return tf2sos(b, a)


def loudest_rms(
    signal: Signal, window: float = 0.1, sr: int = CONFIG.sr
) -> float:
    """The RMS of the loudest ``window`` seconds of the signal."""
    if len(signal) == 0:
        return 0.0
    length = min(round(window * sr), len(signal))
    rms = librosa.feature.rms(
        y=signal, frame_length=length, hop_length=sr // 100, center=False
    )
    return float(rms.max())


def level_signal(signal: Signal, level: float, sr: int = CONFIG.sr) -> Signal:
    """Scale the signal so that its ``loudest_rms`` is ``level``."""
    current = loudest_rms(signal, sr=sr)
    if current == 0:
        return signal
    return (signal * (level / current)).astype(np.float32)


def normalize_signal(signal: Signal) -> Signal:
    return librosa.util.normalize(signal)


def sidechain_signal(
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


def section_signal(
    signal: Signal,
    effect: Effect,
    start: int,
    end: int,
    fade: float = 0.005,
    sr: int = CONFIG.sr,
) -> Signal:
    """Apply the effect only from sample ``start`` to ``end``.

    The effect is applied to the whole signal, so its state is continuous at
    the section edges. The section fades in after ``start`` and out before
    ``end``, which prevents clicks.
    """
    end = min(end, len(signal))
    if not 0 <= start <= end:
        raise ValueError("The section must start inside the signal.")
    wet = effect.process(signal, sr)
    if len(wet) != len(signal):
        raise ValueError("The effect of a section must keep the length.")

    mix = np.zeros(len(signal), dtype=np.float32)
    mix[start:end] = 1.0
    fade_samples = min(round(fade * sr), (end - start) // 2)
    if fade_samples:
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        mix[start : start + fade_samples] = ramp
        mix[end - fade_samples : end] = ramp[::-1]
    return np.asarray(signal + mix * (wet - signal), dtype=np.float32)


@dataclass(frozen=True)
class Cut(Effect):
    length: int
    fade: float = 0.01

    def process(self, signal: Signal, sr: int) -> Signal:
        return cut_signal(signal, self.length, self.fade, sr=sr)


@dataclass(frozen=True)
class Trim(Effect):
    def process(self, signal: Signal, sr: int) -> Signal:
        return trim_signal(signal)


@dataclass(frozen=True)
class Filter(Effect):
    cutoff: float | tuple[float, float]
    btype: str
    order: int = 2
    q: float | None = None

    def process(self, signal: Signal, sr: int) -> Signal:
        return filter_signal(
            signal, self.cutoff, self.btype, self.order, self.q, sr=sr
        )


@dataclass(frozen=True)
class Normalize(Effect):
    def process(self, signal: Signal, sr: int) -> Signal:
        return normalize_signal(signal)


@dataclass(frozen=True, eq=False)
class Sidechain(Effect):
    triggers: np.ndarray = field(repr=False)
    ratio: float = 0.5
    duration: float = 0.1
    fade: float = 0.005

    def process(self, signal: Signal, sr: int) -> Signal:
        return sidechain_signal(
            signal,
            self.triggers,
            self.ratio,
            self.duration,
            self.fade,
            sr=sr,
        )


@dataclass(frozen=True)
class Section(Effect):
    effect: Effect
    start: int
    """First sample of the section."""
    end: int
    """Sample after the section."""
    fade: float = 0.005

    def process(self, signal: Signal, sr: int) -> Signal:
        return section_signal(
            signal, self.effect, self.start, self.end, self.fade, sr=sr
        )
