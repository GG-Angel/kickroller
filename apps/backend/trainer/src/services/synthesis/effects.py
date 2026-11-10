import io
from dataclasses import dataclass, field
from typing import Protocol

import librosa
import numpy as np
import soundfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import bilinear, butter, sosfilt, tf2sos

from src.models.audio import Signal


def trim_to_attack(signal: Signal, top_db: float = 20.0) -> Signal:
    """Trim the signal, and start it at the first sample within ``top_db`` of its peak.

    The -60 dB trim keeps MP3 encoder delay and pre-echo (about 25 ms), which
    would put the onset label before the attack.
    """
    trimmed, _ = librosa.effects.trim(signal)
    magnitude = np.abs(trimmed)
    return trimmed[
        np.argmax(magnitude >= magnitude.max() * 10 ** (-top_db / 20)) :
    ]


def _resonant_sos(
    cutoff: float | tuple[float, float],
    btype: str,
    order: int,
    q: float,
    sr: int,
) -> np.ndarray:
    if not isinstance(cutoff, (int, float)) or not 0 < cutoff < sr / 2:
        raise ValueError("A resonant filter needs one cutoff below Nyquist.")
    if btype not in ("lowpass", "highpass"):
        raise ValueError("A resonant filter must be a lowpass or highpass.")
    if order < 2 or order % 2:
        raise ValueError("A resonant filter must have an even order of 2+.")
    if q <= 0:
        raise ValueError("Q must be positive.")
    resonant = _biquad_sos(cutoff, btype, q, sr)
    if order == 2:
        return resonant
    rest = np.asarray(butter(order - 2, cutoff, btype, fs=sr, output="sos"))
    return np.vstack([resonant, rest])


def _biquad_sos(cutoff: float, btype: str, q: float, sr: int) -> np.ndarray:
    # analog prototype with the cutoff pre-warped for the bilinear transform
    w0 = 2 * sr * np.tan(np.pi * cutoff / sr)
    numerator = [0.0, 0.0, w0**2] if btype == "lowpass" else [1.0, 0.0, 0.0]
    b, a = bilinear(numerator, [1.0, w0 / q, w0**2], fs=sr)
    return tf2sos(b, a)


def loudest_rms(signal: Signal, sr: int, window: float = 0.1) -> float:
    """The RMS of the loudest ``window`` seconds of the signal."""
    if len(signal) == 0:
        return 0.0
    length = min(round(window * sr), len(signal))
    rms = librosa.feature.rms(
        y=signal, frame_length=length, hop_length=sr // 100, center=False
    )
    return float(rms.max())


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
        if len(signal) <= self.length:
            return signal
        fade_samples = min(round(self.fade * sr), self.length)
        cut = signal[: self.length].copy()
        cut[self.length - fade_samples :] *= np.linspace(
            1.0, 0.0, fade_samples, dtype=np.float32
        )
        return cut


@dataclass(frozen=True)
class Trim(Effect):
    def process(self, signal: Signal, sr: int) -> Signal:
        trimmed, _ = librosa.effects.trim(signal)
        return trimmed


@dataclass(frozen=True)
class Filter(Effect):
    cutoff: float | tuple[float, float]
    btype: str
    order: int = 2
    q: float | None = None

    def process(self, signal: Signal, sr: int) -> Signal:
        """Filter with Butterworth, or resonant filtering when ``q`` is set.

        Resonant filters are low/high-pass biquads with a gain of ``q`` at
        cutoff, followed by Butterworth sections for orders above two.
        """
        if self.q is None:
            sos = butter(
                self.order, self.cutoff, self.btype, fs=sr, output="sos"
            )
        else:
            sos = _resonant_sos(self.cutoff, self.btype, self.order, self.q, sr)
        return np.asarray(sosfilt(sos, signal), dtype=np.float32)


@dataclass(frozen=True)
class Normalize(Effect):
    peak_db: float = 0.0

    def process(self, signal: Signal, sr: int) -> Signal:
        peak = float(librosa.db_to_amplitude(self.peak_db))
        return (librosa.util.normalize(signal) * peak).astype(np.float32)


@dataclass(frozen=True)
class Gain(Effect):
    db: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return (signal * float(librosa.db_to_amplitude(self.db))).astype(
            np.float32
        )


@dataclass(frozen=True)
class Level(Effect):
    level: float

    def process(self, signal: Signal, sr: int) -> Signal:
        current = loudest_rms(signal, sr=sr)
        if current == 0:
            return signal
        return (signal * (self.level / current)).astype(np.float32)


@dataclass(frozen=True)
class Tilt(Effect):
    db: float
    pivot: float = 1000.0

    def process(self, signal: Signal, sr: int) -> Signal:
        lows = Filter(self.pivot, "lowpass", order=1).process(signal, sr)
        highs = signal - lows
        return Gain(-self.db / 2).process(lows, sr) + Gain(self.db / 2).process(
            highs, sr
        )


@dataclass(frozen=True)
class Clip(Effect):
    drive_db: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return np.tanh(Gain(self.drive_db).process(signal, sr)).astype(
            np.float32
        )


@dataclass(frozen=True)
class Limit(Effect):
    drive_db: float
    ceiling_db: float = -0.3

    def process(self, signal: Signal, sr: int) -> Signal:
        driven = Gain(self.drive_db).process(signal, sr)
        if len(driven) == 0:
            return driven
        ceiling = float(librosa.db_to_amplitude(self.ceiling_db))
        size = 2 * round(0.01 * sr / 2) + 1
        peaks = maximum_filter1d(np.abs(driven), size=size)
        gain = ceiling / np.maximum(peaks, ceiling)
        gain = uniform_filter1d(gain, size=size, mode="nearest")
        return (driven * gain).astype(np.float32)


@dataclass(frozen=True)
class Mp3(Effect):
    compression_level: float

    def process(self, signal: Signal, sr: int) -> Signal:
        buffer = io.BytesIO()
        soundfile.write(
            buffer,
            signal,
            sr,
            format="MP3",
            subtype="MPEG_LAYER_III",
            compression_level=self.compression_level,
            bitrate_mode="CONSTANT",
        )
        buffer.seek(0)
        decoded, _ = librosa.load(buffer, sr=sr)
        return librosa.util.fix_length(decoded, size=len(signal))


@dataclass(frozen=True, eq=False)
class Sidechain(Effect):
    triggers: np.ndarray = field(repr=False)
    ratio: float = 0.5
    duration: float = 0.1
    fade: float = 0.005

    def process(self, signal: Signal, sr: int) -> Signal:
        _validate_sidechain(self.triggers, self.ratio, self.duration)
        if self.ratio == 0 or self.duration == 0:
            return signal

        samples = len(signal)
        duration_samples = max(1, round(self.duration * sr))
        fade_samples = min(max(1, round(self.fade * sr)), duration_samples // 2)
        attack = 0.5 - 0.5 * np.cos(
            np.pi * np.linspace(0.0, 1.0, fade_samples, endpoint=False)
        )
        release = 0.5 + 0.5 * np.cos(
            np.pi * np.linspace(0.0, 1.0, duration_samples - fade_samples)
        )
        shape = np.concatenate([attack, release]).astype(np.float32)
        envelope = np.zeros(samples, dtype=np.float32)
        for trigger in self.triggers:
            start = int(trigger)
            end = min(start + duration_samples, samples)
            if end <= start:
                continue
            envelope[start:end] = np.maximum(
                envelope[start:end], shape[: end - start]
            )
        return np.asarray(
            signal * (1.0 - self.ratio * envelope), dtype=np.float32
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
        end = min(self.end, len(signal))
        if not 0 <= self.start <= end:
            raise ValueError("The section must start inside the signal.")
        wet = self.effect.process(signal, sr)
        if len(wet) != len(signal):
            raise ValueError("The effect of a section must keep the length.")

        mix = np.zeros(len(signal), dtype=np.float32)
        mix[self.start : end] = 1.0
        fade_samples = min(round(self.fade * sr), (end - self.start) // 2)
        if fade_samples:
            ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            mix[self.start : self.start + fade_samples] = ramp
            mix[end - fade_samples : end] = ramp[::-1]
        return np.asarray(signal + mix * (wet - signal), dtype=np.float32)
