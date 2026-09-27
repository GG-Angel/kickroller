from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyzer.audio import SAMPLE_RATE, TARGET_LUFS, load_mid, normalize_loudness
from analyzer.lowband import low_band_change, low_band_profile
from analyzer.odf import log_filterbank, stft_bins, superflux
from analyzer.peaks import enforce_min_distance, local_peaks


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = SAMPLE_RATE
    target_lufs: float = TARGET_LUFS
    hop: int = 441  # 10 ms
    n_fft: int = 2048  # 46 ms

    # Candidates: attacks in the kick click band.
    click_fmin: float = 2000.0
    click_fmax: float = 6000.0
    bands_per_octave: int = 12
    threshold: float = 1.5
    peak_window: float = 0.02  # local max within +/- 20 ms
    min_distance: float = 0.04

    # Low band (30-300 Hz), compared 30 ms after the attack with 30 ms before.
    low_fmin: float = 30.0
    low_fmax: float = 300.0
    low_context: float = 0.03

    # Path 1: a clear click that starts a new kick body.
    min_low_level_db: float = -35.0  # relative to the track's 95th percentile
    min_low_rise_db: float = 6.0
    min_pitch_jump_hz: float = 12.0

    # Path 2: a weak click where the low band restarts at full level after a gap.
    weak_threshold: float = 0.5
    min_restart_rise_db: float = 12.0
    min_restart_level_db: float = -10.0

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop

    def frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))


def click_strength(signal: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Click-band SuperFlux onset strength of a mono signal (one value per hop)."""
    first_bin, filters = log_filterbank(
        config.n_fft,
        config.sample_rate,
        config.click_fmin,
        config.click_fmax,
        config.bands_per_octave,
    )
    last_bin = first_bin + filters.shape[0] - 1
    magnitude = stft_bins(signal, config.n_fft, config.hop, first_bin, last_bin)
    return superflux(magnitude @ filters)


def detect_kicks_in_signal(signal: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Kick onset times in seconds for a mono signal at `config.sample_rate`."""
    signal = normalize_loudness(signal, config.sample_rate, config.target_lufs)
    strength = click_strength(signal, config)
    candidates = local_peaks(
        strength,
        threshold=min(config.threshold, config.weak_threshold),
        local_max_frames=config.frames(config.peak_window),
    )
    energy, centroid = low_band_profile(
        signal,
        config.sample_rate,
        config.n_fft,
        config.hop,
        config.low_fmin,
        config.low_fmax,
    )
    level_db, rise_db, pitch_jump = low_band_change(
        candidates, energy, centroid, config.frames(config.low_context)
    )

    click = strength[candidates]
    new_body = (level_db >= config.min_low_level_db) & (
        (rise_db >= config.min_low_rise_db) | (pitch_jump >= config.min_pitch_jump_hz)
    )
    restart = (level_db >= config.min_restart_level_db) & (
        rise_db >= config.min_restart_rise_db
    )
    keep = ((click >= config.threshold) & new_body) | (
        (click >= config.weak_threshold) & restart
    )

    kicks = enforce_min_distance(
        candidates[keep], click[keep], config.frames(config.min_distance)
    )
    return kicks / config.fps


def detect_kicks(path: str | Path, config: DetectorConfig | None = None) -> np.ndarray:
    """Kick onset times in seconds for an audio or video file."""
    config = config or DetectorConfig()
    return detect_kicks_in_signal(load_mid(path, config.sample_rate), config)
