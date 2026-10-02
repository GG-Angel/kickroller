"""The detection pipeline: kick model, peak picking and the beat grid."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from loguru import logger

from analyzer.audio.io import (
    SAMPLE_RATE,
    TARGET_LUFS,
    load_mid_channel,
    normalize_loudness,
)
from analyzer.detection.grid import regularize_beats, track_beats
from analyzer.detection.peaks import (
    enforce_min_distance,
    find_local_peaks,
    interpolate_peaks,
)
from analyzer.model.checkpoint import (
    DEFAULT_MODEL,
    load_model,
    predict_kick_probability,
)
from analyzer.model.network import KickNet
from analyzer.settings import DetectorSettings, Settings


@dataclass(frozen=True)
class Detection:
    beats: np.ndarray  # beat times in seconds
    kicks: np.ndarray  # kick onset times in seconds
    confidence: np.ndarray  # confidence (0-1) of each kick


def _select_strongest_per_key(keys: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Index of the strongest item for each distinct key."""
    if len(keys) == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((-strength, keys))
    first = np.r_[True, keys[order][1:] != keys[order][:-1]]
    return order[first]


def _keep_beats_in_track(beats: np.ndarray, duration: float) -> np.ndarray:
    return beats[(beats >= 0.0) & (beats <= duration)]


def _compute_kick_min_distance(grid: np.ndarray, settings: DetectorSettings) -> float:
    """The minimum distance between kicks in seconds, from the tempo of the beat grid.

    The result is at most `settings.min_distance`: a wrong tempo can only make the
    rule shorter. Above about 226 BPM, the beat grid is moved down an octave
    (see `regularize_beats`), so the rule stays at `settings.min_distance`.
    """
    if len(grid) < 2:
        logger.debug(
            "No tempo; the minimum kick distance is {distance:.0f} ms",
            distance=1000 * settings.min_distance,
        )
        return settings.min_distance
    period = float((grid[-1] - grid[0]) / (len(grid) - 1))
    distance = min(settings.min_distance, settings.min_distance_beats * period)
    logger.debug(
        "Minimum kick distance: {distance:.1f} ms ({fraction} of a beat at "
        "{bpm:.1f} BPM, at most {limit:.0f} ms)",
        distance=1000 * distance,
        fraction=settings.min_distance_beats,
        bpm=60.0 / period,
        limit=1000 * settings.min_distance,
    )
    return distance


def _pick_kicks(
    probability: np.ndarray, fps: float, min_distance: float, settings: DetectorSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Kick onset times in seconds and their confidence: the peaks of the kick probability."""
    candidates = find_local_peaks(
        probability,
        threshold=settings.min_confidence,
        local_max_frames=max(1, round(settings.peak_window * fps)),
    )
    distance_frames = max(1, round(min_distance * fps))
    kept = enforce_min_distance(candidates, probability[candidates], distance_frames)
    logger.debug(
        "Kick model: {count} peaks with probability >= {threshold}; removed {removed} "
        "closer than {distance:.0f} ms to a more probable peak",
        count=len(candidates),
        threshold=settings.min_confidence,
        removed=len(candidates) - len(kept),
        distance=1000 * distance_frames / fps,
    )
    times = interpolate_peaks(probability, candidates) / fps
    is_kick = np.zeros(len(candidates), dtype=bool)
    is_kick[kept] = True
    for time, frame, kick in zip(times, candidates, is_kick):
        logger.trace(
            "{time:8.3f} s | probability {probability:.2f} {outcome}",
            time=time,
            probability=probability[frame],
            outcome="kick" if kick else "removed (near a more probable kick)",
        )
    return times[kept], probability[candidates[kept]]


def _align_beats(
    beats: np.ndarray,
    kicks: np.ndarray,
    confidence: np.ndarray,
    settings: DetectorSettings,
) -> np.ndarray:
    """Move each beat to its most confident kick, and the other beats by the median shift.

    beat_this gives beats in 20 ms steps; the kick times are more exact. A beat
    moves only to a kick within `settings.beat_window` with a confidence of at
    least `settings.anchor_confidence`.
    """
    confident = confidence >= settings.anchor_confidence
    times, scores = kicks[confident], confidence[confident]
    if len(beats) == 0 or len(times) == 0:
        logger.warning(
            "No kick with a confidence of at least {threshold}; the beats are not moved",
            threshold=settings.anchor_confidence,
        )
        return beats
    nearest = np.clip(np.searchsorted(beats, times), 1, len(beats) - 1)
    nearest -= (times - beats[nearest - 1]) < (beats[nearest] - times)
    offset = times - beats[nearest]
    near = np.flatnonzero(np.abs(offset) <= settings.beat_window)
    near = near[_select_strongest_per_key(nearest[near], scores[near])]
    if len(near) == 0:
        logger.warning("No confident kick is near a beat; the beats are not moved")
        return beats
    shift = float(np.median(offset[near]))
    moved = beats + shift
    moved[nearest[near]] = times[near]
    logger.debug(
        "Moved {moved} beats to their kicks (confidence >= {threshold}) "
        "and the other {others} by the median shift of {shift:+.1f} ms",
        moved=len(near),
        threshold=settings.anchor_confidence,
        others=len(beats) - len(near),
        shift=1000 * shift,
    )
    return moved


def detect_kicks_in_signal(
    signal: np.ndarray, model: KickNet, settings: DetectorSettings
) -> Detection:
    """Beat times, kick onset times and kick confidence for a mono signal.

    The signal must be at SAMPLE_RATE. The confidence of a kick is the model's
    kick probability. Only kicks with a confidence of at least
    `settings.min_confidence` are returned. The beats only set the minimum distance
    between kicks (from the tempo); each beat is then moved to its confident kick.
    """
    duration = len(signal) / SAMPLE_RATE
    start = perf_counter()
    probability = predict_kick_probability(
        model,
        normalize_loudness(signal, SAMPLE_RATE, TARGET_LUFS),
        settings.chunk_seconds,
    )
    logger.debug("Kick model ran in {seconds:.1f} s", seconds=perf_counter() - start)
    tracked = track_beats(signal, SAMPLE_RATE, settings.beat_checkpoint)
    grid = regularize_beats(
        tracked,
        duration,
        settings.min_bpm,
        settings.max_bpm,
        settings.min_beat_interval,
    )
    min_distance = _compute_kick_min_distance(grid, settings)
    kicks, confidence = _pick_kicks(
        probability, model.settings.fps, min_distance, settings
    )
    if len(kicks) == 0:
        logger.warning("The kick model found no kicks")

    beats = _keep_beats_in_track(
        _align_beats(grid, kicks, confidence, settings), duration
    )
    logger.info(
        "Detected {kicks} kicks ({sure} with confidence >= {threshold}) "
        "and {beats} beats",
        kicks=len(kicks),
        sure=np.count_nonzero(confidence >= settings.anchor_confidence),
        threshold=settings.anchor_confidence,
        beats=len(beats),
    )
    return Detection(beats, kicks, confidence)


def detect_kicks(
    path: str | Path,
    settings: DetectorSettings | None = None,
    model_path: Path = DEFAULT_MODEL,
) -> Detection:
    """Beat times, kick onset times and kick confidence for an audio or video file.

    Without `settings`, the detector settings come from `Settings` (defaults and
    ANALYZER_* environment variables).
    """
    settings = settings or Settings().detector
    model = load_model(model_path)
    return detect_kicks_in_signal(load_mid_channel(path), model, settings)
