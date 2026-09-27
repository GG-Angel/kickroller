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
from analyzer.lowband import low_band_change, low_band_gap_rise, low_band_profile
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

    # Confidence: each rule below is a product of sigmoids centered on its
    # thresholds, and the score is the highest rule score. "Maybe" rules have a
    # cap below 0.5.
    min_confidence: float = 0.1
    anchor_confidence: float = 0.5  # beats move only to kicks with this score
    click_width: float = 0.25  # sigmoid widths: octaves of click strength,
    db_width: float = 2.0  # dB,
    hz_width: float = 3.0  # and Hz

    # Click thresholds per position kind: stricter at the finer positions.
    beat_threshold: float = 0.9
    eighth_threshold: float = 1.5
    fine_threshold: float = 1.5

    # Rule 1, new kick body: a click, and the low band has level and rises or
    # jumps up in pitch. At 1/16 and triplet positions, only a pitch jump counts.
    min_low_level_db: float = -35.0  # relative to the track's 95th percentile
    min_low_rise_db: float = 6.0
    min_pitch_jump_hz: float = 12.0
    fine_min_level_db: float = -20.0

    # Rule 2, restart (beat, 1/8): a weak click where the low band restarts at
    # full level after a gap.
    weak_threshold: float = 0.5
    min_restart_rise_db: float = 12.0
    min_restart_level_db: float = -10.0

    # Rule 3, maybe (beat, 1/8): the low band starts with the click, as in a
    # regular drum kick (short-gap rise).
    min_gap_rise_db: float = 8.0
    min_gap_level_db: float = -25.0
    gap_cap_beat: float = 0.5
    gap_cap_eighth: float = 0.4

    # Rule 4, maybe (beat only): a strong click with no low-band evidence.
    strong_threshold: float = 1.8
    strong_cap: float = 0.3

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop

    def frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))


@dataclass(frozen=True)
class Detection:
    beats: np.ndarray  # beat grid times in seconds
    kicks: np.ndarray  # kick onset times in seconds
    confidence: np.ndarray  # confidence (0-1) of each kick


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


def soft_at_least(x: np.ndarray, threshold: float, width: float) -> np.ndarray:
    """0.5 at `threshold`, near 1 well above it and near 0 well below it."""
    return 0.5 * (1.0 + np.tanh((x - threshold) / (2.0 * width)))


def kick_scores(
    click: np.ndarray,
    level_db: np.ndarray,
    rise_db: np.ndarray,
    pitch_jump: np.ndarray,
    gap_rise_db: np.ndarray,
    config: DetectorConfig,
) -> np.ndarray:
    """Confidence (0-1) of each candidate, one row per position kind.

    Rows: BEAT, EIGHTH, SIXTEENTH, TRIPLET.
    """
    c = config
    log_click = np.log2(np.maximum(click, 1e-9))

    def clicks(threshold: float) -> np.ndarray:
        return soft_at_least(log_click, np.log2(threshold), c.click_width)

    def db(x: np.ndarray, threshold: float) -> np.ndarray:
        return soft_at_least(x, threshold, c.db_width)

    jump = soft_at_least(pitch_jump, c.min_pitch_jump_hz, c.hz_width)
    body = np.maximum(jump, db(rise_db, c.min_low_rise_db)) * db(
        level_db, c.min_low_level_db
    )
    restart = (
        clicks(c.weak_threshold)
        * db(rise_db, c.min_restart_rise_db)
        * db(level_db, c.min_restart_level_db)
    )
    gap_start = (
        clicks(c.weak_threshold)
        * db(gap_rise_db, c.min_gap_rise_db)
        * db(level_db, c.min_gap_level_db)
    )
    beat = np.max(
        [
            clicks(c.beat_threshold) * body,
            restart,
            c.gap_cap_beat * gap_start,
            c.strong_cap * clicks(c.strong_threshold),
        ],
        axis=0,
    )
    eighth = np.max(
        [clicks(c.eighth_threshold) * body, restart, c.gap_cap_eighth * gap_start],
        axis=0,
    )
    fine = clicks(c.fine_threshold) * jump * db(level_db, c.fine_min_level_db)
    return np.stack([beat, eighth, fine, fine])


def strongest_per_key(keys: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Index of the strongest item for each distinct key."""
    if len(keys) == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((-strength, keys))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    return order[first]


def beats_in_track(beats: np.ndarray, duration: float) -> np.ndarray:
    return beats[(beats >= 0.0) & (beats <= duration)]


def detect_kicks_in_signal(signal: np.ndarray, config: DetectorConfig) -> Detection:
    """Beat grid, kick onset times and kick confidence for a mono signal.

    The signal must be at `config.sample_rate`. Only kicks with a confidence of
    at least `config.min_confidence` are returned. The beats are the grid used
    for the kicks: each beat is moved to its confident kick.
    """
    duration = len(signal) / config.sample_rate
    grid = regularize_beats(
        track_beats(signal, config.sample_rate, config.beat_checkpoint),
        duration,
        config.min_bpm,
        config.max_bpm,
    )
    no_kicks = Detection(beats_in_track(grid, duration), np.empty(0), np.empty(0))
    if len(grid) < 2:
        return no_kicks

    signal = normalize_loudness(signal, config.sample_rate, config.target_lufs)
    strength = click_strength(signal, config)
    candidates = local_peaks(
        strength,
        threshold=config.weak_threshold,
        local_max_frames=config.frames(config.peak_window),
    )
    if len(candidates) == 0:
        return no_kicks
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
    scores = kick_scores(
        click,
        level_db,
        rise_db,
        pitch_jump,
        low_band_gap_rise(candidates, energy),
        config,
    )
    times = candidates / config.fps
    windows = np.array(
        [
            config.beat_window,
            config.eighth_window,
            config.fine_window,
            config.fine_window,
        ]
    )

    # 1. Beats: the highest-scoring candidate near each beat.
    interval, position, offset = nearest_position(times, grid)
    near = np.flatnonzero(
        (KINDS[position] == BEAT) & (np.abs(offset) <= config.beat_window)
    )
    beat_index = interval + (position == len(POSITIONS) - 1)
    beat_kicks = near[strongest_per_key(beat_index[near], scores[BEAT, near])]

    # 2. beat_this gives beats in 20 ms steps: move each beat to its confident
    # kick, and move the other beats by the median shift.
    anchors = grid.copy()
    confident = beat_kicks[scores[BEAT, beat_kicks] >= config.anchor_confidence]
    if len(confident):
        kick_beats = beat_index[confident]
        anchors += np.median(times[confident] - grid[kick_beats])
        anchors[kick_beats] = times[confident]

    # 3. Off-beat positions between the moved beats.
    interval, position, offset = nearest_position(times, anchors)
    kind = KINDS[position]
    score = scores[kind, np.arange(len(times))]
    ok = (kind != BEAT) & (np.abs(offset) <= windows[kind])
    ok[beat_kicks] = False
    ok = np.flatnonzero(ok)
    off_beat = ok[
        strongest_per_key(interval[ok] * len(POSITIONS) + position[ok], score[ok])
    ]
    off_beat = off_beat[score[off_beat] >= config.min_confidence]

    # 4. In each beat, keep the straight (1/8, 1/16) or the triplet kicks,
    # whichever has the higher best score.
    triplet = KINDS[position[off_beat]] == TRIPLET
    beat_of = interval[off_beat]
    best = np.zeros((2, len(anchors)))
    np.maximum.at(best, (triplet.astype(int), beat_of), score[off_beat])
    off_beat = off_beat[triplet == (best[1] > best[0])[beat_of]]

    kicks = np.concatenate([beat_kicks, off_beat])
    confidence = np.concatenate([scores[BEAT, beat_kicks], score[off_beat]])
    keep = confidence >= config.min_confidence
    kicks, confidence = kicks[keep], confidence[keep]
    kept = enforce_min_distance(
        candidates[kicks], confidence, config.frames(config.min_distance)
    )
    return Detection(
        beats_in_track(anchors, duration),
        candidates[kicks[kept]] / config.fps,
        confidence[kept],
    )


def detect_kicks(path: str | Path, config: DetectorConfig | None = None) -> Detection:
    """Beat grid, kick onset times and kick confidence for an audio or video file."""
    config = config or DetectorConfig()
    return detect_kicks_in_signal(load_mid(path, config.sample_rate), config)
