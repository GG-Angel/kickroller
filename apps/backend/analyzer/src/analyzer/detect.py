from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from loguru import logger

from analyzer.audio import SAMPLE_RATE, TARGET_LUFS, load_mid, normalize_loudness
from analyzer.grid import (
    BEAT,
    KIND_NAMES,
    KINDS,
    POSITIONS,
    TRIPLET,
    nearest_position,
    regularize_beats,
    track_beats,
)
from analyzer.model import DEFAULT_MODEL, KickNet, kick_activation, load_model
from analyzer.peaks import enforce_min_distance, local_peaks


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = SAMPLE_RATE
    target_lufs: float = TARGET_LUFS

    # Beat grid from beat_this, with 1/16 and triplet positions between the beats.
    beat_checkpoint: str = "final0"
    min_bpm: float = 150.0
    max_bpm: float = 170.0
    beat_window: float = 0.04  # search window around each position
    eighth_window: float = 0.03
    fine_window: float = 0.015  # 1/16 and triplet positions

    # Candidates: peaks of the model's kick probability.
    peak_window: float = 0.02  # local max within +/- 20 ms
    min_distance: float = 0.04
    min_confidence: float = 0.1
    anchor_confidence: float = 0.5  # beats move only to kicks with this confidence


@dataclass(frozen=True)
class Detection:
    beats: np.ndarray  # beat grid times in seconds
    kicks: np.ndarray  # kick onset times in seconds
    confidence: np.ndarray  # confidence (0-1) of each kick


def strongest_per_key(keys: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Index of the strongest item for each distinct key."""
    if len(keys) == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((-strength, keys))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    return order[first]


def beats_in_track(beats: np.ndarray, duration: float) -> np.ndarray:
    return beats[(beats >= 0.0) & (beats <= duration)]


def detect_kicks_in_signal(
    signal: np.ndarray, model: KickNet, config: DetectorConfig
) -> Detection:
    """Beat grid, kick onset times and kick confidence for a mono signal.

    The signal must be at `config.sample_rate`. The confidence of a kick is the
    model's kick probability. Only kicks with a confidence of at least
    `config.min_confidence` are returned. The beats are the grid used for the
    kicks: each beat is moved to its confident kick.
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
        logger.warning("No beat grid; no kicks")
        return no_kicks

    signal = normalize_loudness(signal, config.sample_rate, config.target_lufs)
    fps = model.config.fps
    start = perf_counter()
    probability = kick_activation(model, signal)
    candidates = local_peaks(
        probability,
        threshold=config.min_confidence,
        local_max_frames=max(1, round(config.peak_window * fps)),
    )
    logger.debug(
        "Kick model: {count} peaks with probability >= {threshold} in {seconds:.1f} s",
        count=len(candidates),
        threshold=config.min_confidence,
        seconds=perf_counter() - start,
    )
    if len(candidates) == 0:
        logger.warning("The kick model found no kicks")
        return no_kicks
    score = probability[candidates]
    times = candidates / fps
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
    beat_kicks = near[strongest_per_key(beat_index[near], score[near])]
    logger.debug(
        "Beat pass: {found} of {count} beats have a candidate within {window:.0f} ms",
        found=len(beat_kicks),
        count=len(grid),
        window=1000 * config.beat_window,
    )

    # 2. beat_this gives beats in 20 ms steps: move each beat to its confident
    # kick, and move the other beats by the median shift.
    anchors = grid.copy()
    confident = beat_kicks[score[beat_kicks] >= config.anchor_confidence]
    if len(confident):
        kick_beats = beat_index[confident]
        shift = np.median(times[confident] - grid[kick_beats])
        anchors += shift
        anchors[kick_beats] = times[confident]
        logger.debug(
            "Moved {moved} beats to their kicks (confidence >= {threshold}) "
            "and the other {others} by the median shift of {shift:+.1f} ms",
            moved=len(confident),
            threshold=config.anchor_confidence,
            others=len(grid) - len(confident),
            shift=1000 * shift,
        )
    else:
        logger.warning(
            "No beat kick has a confidence of at least {threshold}; "
            "the beats are not moved",
            threshold=config.anchor_confidence,
        )

    # 3. Off-beat positions between the moved beats.
    interval, position, offset = nearest_position(times, anchors)
    kind = KINDS[position]
    ok = (kind != BEAT) & (np.abs(offset) <= windows[kind])
    ok[beat_kicks] = False
    ok = np.flatnonzero(ok)
    off_beat = ok[
        strongest_per_key(interval[ok] * len(POSITIONS) + position[ok], score[ok])
    ]
    off_beat = off_beat[score[off_beat] >= config.min_confidence]
    logger.debug(
        "Off-beat pass: {count} positions have a kick with confidence >= {threshold}",
        count=len(off_beat),
        threshold=config.min_confidence,
    )

    # 4. In each beat, keep the straight (1/8, 1/16) or the triplet kicks,
    # whichever has the higher best score.
    triplet = KINDS[position[off_beat]] == TRIPLET
    beat_of = interval[off_beat]
    best = np.zeros((2, len(anchors)))
    np.maximum.at(best, (triplet.astype(int), beat_of), score[off_beat])
    chosen = triplet == (best[1] > best[0])[beat_of]
    logger.debug(
        "Straight or triplet: kept {straight} straight and {triplet} triplet kicks, "
        "removed {removed}",
        straight=np.count_nonzero(chosen & ~triplet),
        triplet=np.count_nonzero(chosen & triplet),
        removed=np.count_nonzero(~chosen),
    )
    off_beat = off_beat[chosen]

    kicks = np.concatenate([beat_kicks, off_beat])
    kicks = kicks[np.argsort(times[kicks])]
    kicks = kicks[score[kicks] >= config.min_confidence]
    kept = enforce_min_distance(
        candidates[kicks], score[kicks], max(1, round(config.min_distance * fps))
    )
    logger.debug(
        "Removed {removed} kicks closer than {distance:.0f} ms to a more probable kick",
        removed=len(kicks) - len(kept),
        distance=1000 * config.min_distance,
    )
    kicks = kicks[kept]
    confidence = score[kicks]

    detection = Detection(beats_in_track(anchors, duration), times[kicks], confidence)
    on_beat = np.count_nonzero(np.isin(kicks, beat_kicks))
    logger.info(
        "Detected {kicks} kicks ({on_beat} on beats, {off_beat} off-beat; "
        "{sure} with confidence >= {threshold}) and {beats} beats",
        kicks=len(kicks),
        on_beat=on_beat,
        off_beat=len(kicks) - on_beat,
        sure=np.count_nonzero(confidence >= config.anchor_confidence),
        threshold=config.anchor_confidence,
        beats=len(detection.beats),
    )

    # One line per candidate, for tuning: its nearest grid position and probability.
    on_grid = np.abs(offset) <= windows[kind]
    outcome = np.where(on_grid, "rejected", "off grid").astype(object)
    outcome[kicks] = "kick"
    for i, frame in enumerate(candidates):
        logger.trace(
            "{time:8.3f} s {position:<7} {offset:+6.1f} ms | "
            "probability {probability:.2f} {outcome}",
            time=frame / fps,
            position=KIND_NAMES[kind[i]],
            offset=1000 * offset[i],
            probability=score[i],
            outcome=outcome[i],
        )
    return detection


def detect_kicks(
    path: str | Path,
    config: DetectorConfig | None = None,
    model_path: Path = DEFAULT_MODEL,
) -> Detection:
    """Beat grid, kick onset times and kick confidence for an audio or video file."""
    config = config or DetectorConfig()
    model = load_model(model_path)
    return detect_kicks_in_signal(load_mid(path, config.sample_rate), model, config)
