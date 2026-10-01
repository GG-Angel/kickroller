"""The beat grid from beat_this."""

from time import perf_counter

import numpy as np
from beat_this.inference import Audio2Beats
from loguru import logger

MIN_SIGNAL_SECONDS = 1.0  # beat_this fails on shorter input


def track_beats(
    signal: np.ndarray, sample_rate: int, checkpoint: str = "final0"
) -> np.ndarray:
    """Beat times in seconds from the beat_this model (downloaded on first use).

    Returns no beats for signals shorter than MIN_SIGNAL_SECONDS.
    """
    if len(signal) < MIN_SIGNAL_SECONDS * sample_rate:
        logger.warning(
            "Signal is shorter than {seconds:g} s; no beats", seconds=MIN_SIGNAL_SECONDS
        )
        return np.empty(0)
    logger.debug(
        "Tracking beats with beat_this ({checkpoint} model, CPU)", checkpoint=checkpoint
    )
    start = perf_counter()
    beats, _ = Audio2Beats(checkpoint_path=checkpoint, device="cpu", dbn=False)(
        signal, sample_rate
    )
    logger.debug(
        "beat_this found {count} beats in {seconds:.1f} s",
        count=len(beats),
        seconds=perf_counter() - start,
    )
    return np.asarray(beats, dtype=float)


def regularize_beats(
    beats: np.ndarray,
    duration: float,
    min_bpm: float,
    max_bpm: float,
    min_interval: float,
) -> np.ndarray:
    """Remove extra beats, fill skipped beats and extend the beats over the full track.

    The beat period is the median beat interval, moved by octaves to the
    `min_bpm`-`max_bpm` range. A beat closer than `min_interval` beat periods to
    the beat before it is extra. The result starts at or before 0 s and ends at
    or after `duration`. Returns the beats unchanged if there are fewer than 2.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    if len(beats) < 2:
        logger.warning("Fewer than 2 beats; no beat grid")
        return beats
    period = float(np.median(np.diff(beats)))
    target = 60.0 / np.sqrt(min_bpm * max_bpm)
    octaves = round(np.log2(target / period))
    if octaves:
        logger.debug(
            "Median beat interval is {bpm:.1f} BPM; "
            "moved by {octaves:+d} octave(s) to {folded:.1f} BPM",
            bpm=60.0 / period,
            octaves=octaves,
            folded=60.0 / (period * 2.0**octaves),
        )
    period *= 2.0**octaves

    kept = [beats[0]]
    for beat in beats[1:]:
        if beat - kept[-1] >= min_interval * period:
            kept.append(beat)

    filled = [kept[0]]
    for beat in kept[1:]:
        start = filled[-1]
        count = max(1, round((beat - start) / period))
        filled.extend(start + (beat - start) * np.arange(1, count + 1) / count)
    filled = np.array(filled)

    if len(filled) > 1:
        period = (filled[-1] - filled[0]) / (len(filled) - 1)
    before = filled[0] - period * np.arange(int(np.ceil(filled[0] / period)), 0, -1)
    after = filled[-1] + period * np.arange(
        1, int(np.ceil((duration - filled[-1]) / period)) + 1
    )
    logger.info(
        "Beat grid: {bpm:.2f} BPM, {count} beats ({removed} extra beats removed, "
        "{filled} skipped beats filled, {edges} added at the edges)",
        bpm=60.0 / period,
        count=len(filled) + len(before) + len(after),
        removed=len(beats) - len(kept),
        filled=len(filled) - len(kept),
        edges=len(before) + len(after),
    )
    return np.concatenate([before, filled, after])
