import numpy as np
from beat_this.inference import Audio2Beats

BEAT, EIGHTH, SIXTEENTH, TRIPLET = range(4)

# Grid positions inside one beat, as a fraction of the beat, and their kinds.
POSITIONS = np.array([0.0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1.0])
KINDS = np.array([BEAT, SIXTEENTH, TRIPLET, EIGHTH, TRIPLET, SIXTEENTH, BEAT])


def track_beats(
    signal: np.ndarray, sample_rate: int, checkpoint: str = "final0"
) -> np.ndarray:
    """Beat times in seconds from the beat_this model (downloaded on first use).

    Returns no beats for signals shorter than 1 s (beat_this fails on very short input).
    """
    if len(signal) < sample_rate:
        return np.empty(0)
    beats, _ = Audio2Beats(checkpoint_path=checkpoint, device="cpu", dbn=False)(
        signal, sample_rate
    )
    return np.asarray(beats, dtype=float)


def regularize_beats(
    beats: np.ndarray, duration: float, min_bpm: float, max_bpm: float
) -> np.ndarray:
    """Remove extra beats, fill skipped beats and extend the beats over the full track.

    The beat period is the median beat interval, moved by octaves to the
    `min_bpm`-`max_bpm` range. The result starts at or before 0 s and ends at or
    after `duration`. Returns the beats unchanged if there are fewer than 2.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    if len(beats) < 2:
        return beats
    period = float(np.median(np.diff(beats)))
    target = 60.0 / np.sqrt(min_bpm * max_bpm)
    period *= 2.0 ** round(np.log2(target / period))

    kept = [beats[0]]
    for beat in beats[1:]:
        if beat - kept[-1] >= 0.75 * period:
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
    return np.concatenate([before, filled, after])


def nearest_position(
    times: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest grid position for each time.

    Returns (interval, position, offset): the index of the beat interval
    [grid[i], grid[i + 1]), the index into POSITIONS, and the time minus the
    position time in seconds. `grid` must have at least 2 beats.
    """
    interval = np.clip(np.searchsorted(grid, times, side="right") - 1, 0, len(grid) - 2)
    length = grid[interval + 1] - grid[interval]
    fraction = (times - grid[interval]) / length
    position = np.abs(fraction[:, None] - POSITIONS).argmin(axis=1)
    return interval, position, (fraction - POSITIONS[position]) * length
