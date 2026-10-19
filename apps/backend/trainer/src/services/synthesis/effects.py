import io
from dataclasses import dataclass, field
from typing import Protocol

import librosa
import numpy as np
import soundfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d
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
    """Filter with a Butterworth filter, or with a resonant filter if ``q`` is set.

    The resonant filter is a low-pass or high-pass of an even order, with
    6 dB/oct of slope for each order. It is one resonant biquad, with a gain of
    ``q`` at the cutoff, followed by a Butterworth filter of ``order - 2``. Thus
    the gain at the cutoff is ``q`` for order 2 and ``q/√2`` for higher orders.
    A ``q`` of ``1/√2`` gives no resonance.
    """
    if q is None:
        sos = butter(order, cutoff, btype, fs=sr, output="sos")
    else:
        sos = _resonant_sos(cutoff, btype, order, q, sr)
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


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
    if not q > 0:
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


def normalize_signal(signal: Signal, peak_db: float = 0.0) -> Signal:
    """Scale the signal so that its peak is ``peak_db`` dBFS."""
    peak = float(librosa.db_to_amplitude(peak_db))
    return (librosa.util.normalize(signal) * peak).astype(np.float32)


def gain_signal(signal: Signal, db: float) -> Signal:
    return (signal * float(librosa.db_to_amplitude(db))).astype(np.float32)


def tilt_signal(
    signal: Signal, db: float, pivot: float = 1000.0, sr: int = CONFIG.sr
) -> Signal:
    """Tilt the spectrum: the highs get ``db/2`` and the lows ``-db/2`` of gain.

    The lows are a 1st-order low-pass at the pivot, and the highs are the rest.
    Thus a tilt of 0 dB gives the same signal.
    """
    lows = filter_signal(signal, pivot, "lowpass", order=1, sr=sr)
    highs = signal - lows
    return gain_signal(lows, -db / 2) + gain_signal(highs, db / 2)


def clip_signal(signal: Signal, drive_db: float) -> Signal:
    """Soft clip with ``tanh`` after a gain of ``drive_db``."""
    return np.tanh(gain_signal(signal, drive_db)).astype(np.float32)


def limit_signal(
    signal: Signal,
    drive_db: float,
    ceiling_db: float = -0.3,
    window: float = 0.01,
    sr: int = CONFIG.sr,
) -> Signal:
    """A look-ahead limiter after a gain of ``drive_db``.

    The gain at each sample is the lowest gain that the peaks in the centered
    ``window`` need, smoothed with a mean over the same window. Thus the peaks
    do not go above ``ceiling_db`` and the signal is not delayed.
    """
    driven = gain_signal(signal, drive_db)
    if len(driven) == 0:
        return driven
    ceiling = float(librosa.db_to_amplitude(ceiling_db))
    size = 2 * round(window * sr / 2) + 1
    peaks = maximum_filter1d(np.abs(driven), size=size)
    gain = ceiling / np.maximum(peaks, ceiling)
    gain = uniform_filter1d(gain, size=size, mode="nearest")
    return (driven * gain).astype(np.float32)


def encode_mp3_signal(
    signal: Signal, compression_level: float, sr: int = CONFIG.sr
) -> Signal:
    """Encode the signal to a constant-bitrate MP3 and decode it again.

    A ``compression_level`` of 0 is about 320 kbps and 0.75 is about 96 kbps.
    """
    buffer = io.BytesIO()
    soundfile.write(
        buffer,
        signal,
        sr,
        format="MP3",
        subtype="MPEG_LAYER_III",
        compression_level=compression_level,
        bitrate_mode="CONSTANT",
    )
    buffer.seek(0)
    decoded, _ = librosa.load(buffer, sr=sr)
    return librosa.util.fix_length(decoded, size=len(signal))


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
    peak_db: float = 0.0

    def process(self, signal: Signal, sr: int) -> Signal:
        return normalize_signal(signal, self.peak_db)


@dataclass(frozen=True)
class Gain(Effect):
    db: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return gain_signal(signal, self.db)


@dataclass(frozen=True)
class Tilt(Effect):
    db: float
    pivot: float = 1000.0

    def process(self, signal: Signal, sr: int) -> Signal:
        return tilt_signal(signal, self.db, self.pivot, sr=sr)


@dataclass(frozen=True)
class Clip(Effect):
    drive_db: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return clip_signal(signal, self.drive_db)


@dataclass(frozen=True)
class Limit(Effect):
    drive_db: float
    ceiling_db: float = -0.3

    def process(self, signal: Signal, sr: int) -> Signal:
        return limit_signal(signal, self.drive_db, self.ceiling_db, sr=sr)


@dataclass(frozen=True)
class Mp3(Effect):
    compression_level: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return encode_mp3_signal(signal, self.compression_level, sr=sr)


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
