"""Mastering of synthetic drops: EQ tilt, drive into a soft clipper, and a limiter.

Rawstyle masters are loud and clipped, so the kicks in training must sound like
that too. Each drop gets random settings, and some drops skip the EQ tilt, the
soft clipper or the limiter. Mastering does not move the kick onsets.
"""

import librosa
import numpy as np
import pyloudnorm
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfilt

from core.settings import SETTINGS

SAMPLE_RATE = SETTINGS.sample_rate
SYNTH = SETTINGS.synth

_METER = pyloudnorm.Meter(SAMPLE_RATE)


def _tilt_eq(rng: np.random.Generator, audio: np.ndarray) -> np.ndarray:
    """A low and a high band, each with a random gain."""
    sos = butter(
        SYNTH.master_eq_tilt_filter_order,
        rng.uniform(*SYNTH.master_eq_tilt_corner_hz_range),
        btype="lowpass",
        fs=SAMPLE_RATE,
        output="sos",
    )
    low = sosfilt(sos, audio)
    low_gain, high_gain = librosa.db_to_amplitude(
        rng.uniform(*SYNTH.master_eq_tilt_gain_db_range, size=2)
    )
    return low_gain * low + high_gain * (audio - low)


def _normalize_loudness(audio: np.ndarray, loudness_lufs: float) -> np.ndarray:
    """`audio` scaled to `loudness_lufs` integrated loudness (unchanged if silent)."""
    loudness = _METER.integrated_loudness(audio)
    if not np.isfinite(loudness):
        return audio
    return audio * librosa.db_to_amplitude(loudness_lufs - loudness)


def _limit_peaks(audio: np.ndarray) -> np.ndarray:
    """A peak limiter: a smooth gain that keeps the peaks near the ceiling."""
    window = round(SYNTH.master_limiter_window_seconds * SAMPLE_RATE)
    envelope = maximum_filter1d(np.abs(audio), window)
    gain = np.minimum(1.0, SYNTH.master_limiter_ceiling / np.maximum(envelope, 1e-9))
    return audio * uniform_filter1d(gain, window)


def master_drop(rng: np.random.Generator, audio: np.ndarray) -> np.ndarray:
    """`audio` mastered with random settings, then clipped at full scale."""
    if rng.random() < SYNTH.master_eq_tilt_probability:
        audio = _tilt_eq(rng, audio)
    drive_db = rng.uniform(*SYNTH.master_drive_db_range)
    audio = _normalize_loudness(audio, SYNTH.master_reference_loudness_lufs)
    audio = audio * librosa.db_to_amplitude(drive_db)
    if rng.random() < SYNTH.master_soft_clipper_probability:
        audio = np.tanh(audio)
    if rng.random() < SYNTH.master_limiter_probability:
        audio = _limit_peaks(audio)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
