from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyzer.audio import SAMPLE_RATE, TARGET_LUFS, load_mid, normalize_loudness
from analyzer.grid import (
    BEAT,
    KINDS,
    POSITIONS,
    TRIPLET,
    nearest_position,
    regularize_beats,
    track_beats,
)
from analyzer.lowband import low_band_change, low_band_profile
from analyzer.odf import log_filterbank, stft_bins, superflux
from analyzer.peaks import enforce_min_distance, local_peaks


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = SAMPLE_RATE
    target_lufs: float = TARGET_LUFS
    hop: int = 441  # 10 ms
    n_fft: int = 2048  # 46 ms

    # Beat grid from beat_this, with 1/16 and triplet positions between the beats.
    beat_checkpoint: str = "final0"
    min_bpm: float = 150.0
    max_bpm: float = 170.0
    beat_window: float = 0.04  # search window around each position
    eighth_window: float = 0.03
    fine_window: float = 0.015  # 1/16 and triplet positions

    # Candidates: attacks in the kick click band.
    click_fmin: float = 2000.0
    click_fmax: float = 6000.0
    bands_per_octave: int = 12
    peak_window: float = 0.02  # local max within +/- 20 ms
    min_distance: float = 0.04

    # Low band (30-300 Hz), compared 30 ms after the attack with 30 ms before.
    low_fmin: float = 30.0
    low_fmax: float = 300.0
    low_context: float = 0.03

    # New kick body: the low band has level, and rises or jumps up in pitch.
    min_low_level_db: float = -35.0  # relative to the track's 95th percentile
    min_low_rise_db: float = 6.0
    min_pitch_jump_hz: float = 12.0

    # Restart: a weak click where the low band restarts at full level after a gap.
    weak_threshold: float = 0.5
    min_restart_rise_db: float = 12.0
    min_restart_level_db: float = -10.0

    # Click thresholds per position kind: stricter at the finer positions.
    beat_threshold: float = 0.9
    eighth_threshold: float = 1.5
    fine_threshold: float = 1.5
    fine_min_level_db: float = -20.0

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


def strongest_per_key(keys: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Index of the strongest item for each distinct key."""
    if len(keys) == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((-strength, keys))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    return order[first]


def detect_kicks_in_signal(signal: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Kick onset times in seconds for a mono signal at `config.sample_rate`."""
    grid = regularize_beats(
        track_beats(signal, config.sample_rate, config.beat_checkpoint),
        len(signal) / config.sample_rate,
        config.min_bpm,
        config.max_bpm,
    )
    if len(grid) < 2:
        return np.empty(0)

    signal = normalize_loudness(signal, config.sample_rate, config.target_lufs)
    strength = click_strength(signal, config)
    candidates = local_peaks(
        strength,
        threshold=config.weak_threshold,
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
    times = candidates / config.fps

    new_body = (level_db >= config.min_low_level_db) & (
        (rise_db >= config.min_low_rise_db) | (pitch_jump >= config.min_pitch_jump_hz)
    )
    restart = (
        (click >= config.weak_threshold)
        & (level_db >= config.min_restart_level_db)
        & (rise_db >= config.min_restart_rise_db)
    )
    fine = (
        (click >= config.fine_threshold)
        & (level_db >= config.fine_min_level_db)
        & (pitch_jump >= config.min_pitch_jump_hz)
    )
    passes = np.stack(  # one row per position kind: BEAT, EIGHTH, SIXTEENTH, TRIPLET
        [
            ((click >= config.beat_threshold) & new_body) | restart,
            ((click >= config.eighth_threshold) & new_body) | restart,
            fine,
            fine,
        ]
    )
    windows = np.array(
        [
            config.beat_window,
            config.eighth_window,
            config.fine_window,
            config.fine_window,
        ]
    )

    # 1. Beats: the strongest passing candidate near each beat.
    interval, position, offset = nearest_position(times, grid)
    on_beat = (
        (KINDS[position] == BEAT)
        & (np.abs(offset) <= config.beat_window)
        & passes[BEAT]
    )
    beat_index = interval + (position == len(POSITIONS) - 1)
    on_beat = np.flatnonzero(on_beat)
    beat_kicks = on_beat[strongest_per_key(beat_index[on_beat], click[on_beat])]

    # 2. beat_this gives beats in 20 ms steps: move each beat to its kick, and
    # move the beats with no kick by the median shift.
    anchors = grid.copy()
    if len(beat_kicks):
        kick_beats = beat_index[beat_kicks]
        anchors += np.median(times[beat_kicks] - grid[kick_beats])
        anchors[kick_beats] = times[beat_kicks]

    # 3. Off-beat positions between the moved beats.
    interval, position, offset = nearest_position(times, anchors)
    kind = KINDS[position]
    ok = (
        (kind != BEAT)
        & (np.abs(offset) <= windows[kind])
        & passes[kind, np.arange(len(times))]
    )
    ok[beat_kicks] = False
    ok = np.flatnonzero(ok)
    off_beat = ok[
        strongest_per_key(interval[ok] * len(POSITIONS) + position[ok], click[ok])
    ]

    # 4. In each beat, keep the straight (1/8, 1/16) or the triplet kicks,
    # whichever has the larger total click strength.
    triplet = KINDS[position[off_beat]] == TRIPLET
    beats = interval[off_beat]
    straight_sum = np.bincount(
        beats[~triplet], click[off_beat][~triplet], minlength=len(anchors)
    )
    triplet_sum = np.bincount(
        beats[triplet], click[off_beat][triplet], minlength=len(anchors)
    )
    off_beat = off_beat[triplet == (triplet_sum > straight_sum)[beats]]

    kicks = np.concatenate([beat_kicks, off_beat])
    frames = enforce_min_distance(
        candidates[kicks], click[kicks], config.frames(config.min_distance)
    )
    return frames / config.fps


def detect_kicks(path: str | Path, config: DetectorConfig | None = None) -> np.ndarray:
    """Kick onset times in seconds for an audio or video file."""
    config = config or DetectorConfig()
    return detect_kicks_in_signal(load_mid(path, config.sample_rate), config)
