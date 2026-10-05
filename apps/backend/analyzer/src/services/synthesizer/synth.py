"""Synthetic rawstyle drops with exact kick onset labels.

A drop is rendered on a bar grid that starts up to one bar before the drop:

- Kicks: one kick sample. Each beat can have a kick, a 1/8 off-beat kick or a
    syncopated kick. Some bars end with a kickroll (1/16 notes, 1/8 triplets or
    1/8 notes), more often at the end of a phrase. Each kick cuts the tail of
    the kick before it.
- Loops: stretched to the drop tempo, high-passed and ducked at each kick.
- Claps on beats 2 and 4, random hits on the 1/16 grid, and sometimes an impact.

The level of each sample is the RMS of its loudest part, relative to the kick.
Last, the drop is mastered (EQ tilt, clipping and limiting). A fixed part of the
kicks is held out, for validation drops only.
"""

import warnings
import zlib
from dataclasses import dataclass
from math import ceil

import librosa
import numpy as np
from loguru import logger
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

from core.settings import SETTINGS

from .mastering import master_drop
from .models import Bank, LabeledDrop, Sample

SAMPLE_RATE = SETTINGS.sample_rate
SYNTH = SETTINGS.synth

warnings.filterwarnings(
    "ignore",
    message="The `(hop_length|n_fft)` parameter is deprecated",
    category=FutureWarning,
    module="librosa",
)


@dataclass(frozen=True)
class Grid:
    """
    Positions are in samples from the first bar, which starts `offset` samples
    before the drop. The grid has whole bars and contains the drop, so that no
    position is negative.
    """

    bpm: int
    phase: float  # where the drop starts in the first bar (0 to 1)

    @property
    def beat(self) -> float:
        """The length of a beat in samples."""
        return 60.0 * SAMPLE_RATE / self.bpm

    @property
    def bar(self) -> float:
        """The length of a bar in samples."""
        return SYNTH.beats_per_bar * self.beat

    @property
    def offset(self) -> int:
        """The offset of the drop in samples from the first bar."""
        return int(self.phase * self.bar)

    @property
    def bars(self) -> int:
        """The number of bars that cover the drop including the offset."""
        drop_length = round(SYNTH.drop_duration_seconds * SAMPLE_RATE)
        return ceil((self.offset + drop_length) / self.bar)

    @property
    def length(self) -> int:
        """The length of the drop in samples including the offset."""
        return ceil(self.bars * self.bar)

    def to_samples(self, beats: np.ndarray) -> np.ndarray:
        """Positions in beats as positions in samples."""
        return librosa.time_to_samples(beats * 60.0 / self.bpm, sr=SAMPLE_RATE)

    def to_drop_seconds(self, position: float) -> float:
        """A position in samples as seconds from the drop start (negative: before it)."""
        return (position - self.offset) / SAMPLE_RATE


@dataclass(frozen=True)
class KickPattern:
    """Kick positions in beats from the first bar, in time order."""

    positions: np.ndarray
    in_roll: np.ndarray  # whether each kick is part of a kickroll


def _compute_level(sound: np.ndarray) -> float:
    """The RMS level of the loudest `level_window_seconds` of a sound."""
    window = round(SYNTH.level_window_seconds * SAMPLE_RATE)
    return float(np.sqrt(uniform_filter1d(np.square(sound), window).max()))


def _relative_gain(level_db: float, reference: float, level: float) -> float:
    """The gain that puts a sound at `level` at `level_db` relative to `reference`."""
    return float(librosa.db_to_amplitude(level_db)) * reference / max(level, 1e-9)


def _mix_into(out: np.ndarray, sound: np.ndarray, start: int, gain: float) -> None:
    """Add `gain * sound` to `out` from sample `start`. The part after the end is lost."""
    sound = sound[: max(len(out) - start, 0)]
    out[start : start + len(sound)] += gain * sound


def _cut_tail(sound: np.ndarray, length: int) -> np.ndarray:
    """`sound` cut to `length` samples with a short fade-out (unchanged if it fits)."""
    if len(sound) <= length:
        return sound
    fade = min(round(SYNTH.kick_tail_fade_seconds * SAMPLE_RATE), length)
    cut = sound[:length].copy()
    cut[length - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return cut


def _high_pass(signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """High-pass filter the signal at `cutoff_hz`."""
    sos = butter(
        SYNTH.loop_high_pass_filter_order,
        cutoff_hz,
        btype="highpass",
        fs=SAMPLE_RATE,
        output="sos",
    )
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


def _fit_loop_to_grid(sample: Sample, grid: Grid, high_pass_hz: float) -> np.ndarray:
    """A loop stretched to the drop tempo, high-passed and repeated over the grid."""
    loop = sample.audio
    if sample.bpm != grid.bpm:
        loop = librosa.effects.time_stretch(loop, rate=grid.bpm / sample.bpm)
    bars = max(1, round(len(loop) / grid.bar))
    loop = librosa.util.fix_length(loop, size=round(bars * grid.bar))
    loop = _high_pass(loop, high_pass_hz)
    return np.resize(loop, grid.length)


def _make_sidechain_curve(
    rng: np.random.Generator, starts: np.ndarray, length: int
) -> np.ndarray:
    """A gain curve that dips at each kick start and recovers exponentially."""
    if not len(starts):
        return np.ones(length, dtype=np.float32)
    depth = rng.uniform(*SYNTH.loop_duck_depth_range)
    release = rng.uniform(*SYNTH.loop_duck_release_seconds_range) * SAMPLE_RATE
    positions = np.arange(length)
    previous = np.searchsorted(starts, positions, side="right") - 1
    elapsed = positions - starts[np.maximum(previous, 0)]
    curve = 1.0 - depth * np.exp(-elapsed / release)
    return np.where(previous >= 0, curve, 1.0).astype(np.float32)


def _trim_silence(sound: np.ndarray) -> np.ndarray:
    """Trim silence from the start and end of the sound."""
    trimmed, _ = librosa.effects.trim(
        sound,
        top_db=SYNTH.one_shot_silence_trim_threshold_db,
        frame_length=SYNTH.one_shot_trim_frame_samples,
        hop_length=SYNTH.one_shot_trim_hop_samples,
    )
    return trimmed


def _draw_weighted[T](
    rng: np.random.Generator, table: tuple[tuple[T, float], ...]
) -> T:
    """A value from (value, weight) pairs, drawn in proportion to its weight."""
    weights = np.array([weight for _, weight in table])
    return table[int(rng.choice(len(table), p=weights / weights.sum()))][0]


def _draw_sample(rng: np.random.Generator, samples: list[Sample]) -> Sample:
    """Draw a random sample from the list of samples."""
    return samples[int(rng.integers(len(samples)))]


def _draw_grid(rng: np.random.Generator) -> Grid:
    """Draw a random grid with a BPM chosen according to the configured weights."""
    bpm = _draw_weighted(rng, SYNTH.bpm_probability_weights)
    return Grid(bpm=bpm, phase=rng.random())


def _draw_kickroll_length(rng: np.random.Generator, bar: int) -> int:
    """The number of beats at the end of `bar` that are a kickroll (0: no kickroll)."""
    phrase_end = bar % SYNTH.bars_per_phrase == SYNTH.bars_per_phrase - 1
    probability = (
        SYNTH.phrase_end_kickroll_probability
        if phrase_end
        else SYNTH.kickroll_probability_per_bar
    )
    if rng.random() >= probability:
        return 0
    return _draw_weighted(rng, SYNTH.kickroll_length_beats_weights)


def _draw_kickroll(
    rng: np.random.Generator, first_beat: int, beats: int
) -> list[float]:
    """A kickroll of `beats` beats from `first_beat`, with a kick on each step."""
    kicks_per_beat = _draw_weighted(rng, SYNTH.kickroll_kicks_per_beat_weights)
    first_step = 1 if rng.random() < SYNTH.late_kickroll_probability else 0
    steps = np.arange(first_step, beats * kicks_per_beat)
    return (first_beat + steps / kicks_per_beat).tolist()


def _draw_beat_kicks(
    rng: np.random.Generator,
    beat: int,
    kickless_beat_probability: float,
    off_beat_kick_probability: float,
) -> list[float]:
    """The kicks of one beat: on the beat, then an off-beat or a syncopated kick."""
    kicks: list[float] = []
    if rng.random() >= kickless_beat_probability:
        kicks.append(beat)
    if rng.random() < off_beat_kick_probability:
        kicks.append(beat + 0.5)
    elif rng.random() < SYNTH.syncopated_kick_probability:
        kicks.append(beat + float(rng.choice(SYNTH.syncopated_kick_offsets_beats)))
    return kicks


def _draw_kick_pattern(rng: np.random.Generator, bars: int) -> KickPattern:
    """The kicks of the drop. Some bars have no kicks, and some end with a kickroll."""
    beat_kicks: list[float] = []
    roll_kicks: list[float] = []
    if rng.random() >= SYNTH.kickless_drop_probability:
        kickless_beat_probability = rng.uniform(*SYNTH.kickless_beat_probability_range)
        off_beat_kick_probability = rng.uniform(*SYNTH.off_beat_kick_probability_range)
        for bar in range(bars):
            if rng.random() < SYNTH.kickless_bar_probability:
                continue
            first_beat = bar * SYNTH.beats_per_bar
            roll_beats = _draw_kickroll_length(rng, bar)
            roll_start = first_beat + SYNTH.beats_per_bar - roll_beats
            for beat in range(first_beat, roll_start):
                beat_kicks += _draw_beat_kicks(
                    rng, beat, kickless_beat_probability, off_beat_kick_probability
                )
            if roll_beats:
                roll_kicks += _draw_kickroll(rng, roll_start, roll_beats)

    positions = np.array(beat_kicks + roll_kicks, dtype=float)
    in_roll = np.arange(len(positions)) >= len(beat_kicks)
    order = np.argsort(positions, kind="stable")
    return KickPattern(positions=positions[order], in_roll=in_roll[order])


def _is_held_out(kick: Sample) -> bool:
    """A fixed `held_out_kick_fraction` of the kicks, chosen by name."""
    return zlib.crc32(kick.name.encode()) / 2**32 < SYNTH.held_out_kick_fraction


def _draw_kick(rng: np.random.Generator, kicks: list[Sample], held_out: bool) -> Sample:
    """A random kick from the held-out kicks, or from the other kicks."""
    candidates = [kick for kick in kicks if _is_held_out(kick) == held_out]
    if not candidates:
        kind = "held-out" if held_out else "training"
        raise ValueError(f"The bank has no {kind} kicks")
    return _draw_sample(rng, candidates)


def _render_kicks(
    rng: np.random.Generator,
    kick: np.ndarray,
    starts: np.ndarray,
    in_roll: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    """The kick track: each kick plays until the next kick starts."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not len(starts):
        return out

    levels_db = np.where(
        in_roll,
        rng.uniform(*SYNTH.kickroll_level_db_range, size=len(starts)),
        rng.uniform(*SYNTH.regular_kick_level_db_range, size=len(starts)),
    )
    ends = np.append(starts[1:], grid.length)
    for start, end, level_db in zip(starts, ends, levels_db, strict=True):
        gain = float(librosa.db_to_amplitude(level_db))
        _mix_into(out, _cut_tail(kick, end - start), start, gain)
    return out


def _render_loops(
    rng: np.random.Generator,
    loops: list[Sample],
    starts: np.ndarray,
    grid: Grid,
    reference: float,
) -> np.ndarray:
    """The loop track: a random number of loops, ducked at each kick."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not loops:
        return out
    for _ in range(_draw_weighted(rng, SYNTH.loop_count_weights)):
        sample = _draw_sample(rng, loops)
        high_pass_hz = rng.uniform(*SYNTH.loop_high_pass_cutoff_hz_range)
        loop = _fit_loop_to_grid(sample, grid, high_pass_hz)
        level_db = rng.uniform(*SYNTH.loop_level_db_range)
        logger.debug(
            "Loop {path}: high-passed at {high_pass_hz:.0f} Hz, {level_db:+.1f} dB",
            path=sample.name,
            high_pass_hz=high_pass_hz,
            level_db=level_db,
        )
        out += _relative_gain(level_db, reference, _compute_level(loop)) * loop
    return out * _make_sidechain_curve(rng, starts, grid.length)


def _render_claps(
    rng: np.random.Generator, claps: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The clap track (in some drops): one clap sample on beats 2 and 4."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not claps or rng.random() >= SYNTH.clap_layer_probability:
        return out
    sample = _draw_sample(rng, claps)
    clap = _trim_silence(sample.audio)
    level_db = rng.uniform(*SYNTH.clap_level_db_range)
    logger.debug(
        "Clap {path} on beats 2 and 4: {level_db:+.1f} dB",
        path=sample.name,
        level_db=level_db,
    )
    gain = _relative_gain(level_db, reference, _compute_level(clap))
    for start in grid.to_samples(np.arange(1, grid.bars * SYNTH.beats_per_bar, 2)):
        _mix_into(out, clap, start, gain)
    return out


def _render_hits(
    rng: np.random.Generator, hits: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The hit track: random one-shots on random 1/16 notes."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not hits:
        return out
    sixteenths = grid.bars * SYNTH.beats_per_bar * SYNTH.sixteenth_notes_per_beat
    positions = rng.integers(
        0, sixteenths, size=rng.poisson(SYNTH.mean_random_hits_per_drop)
    )
    for start in grid.to_samples(positions / SYNTH.sixteenth_notes_per_beat):
        sample = _draw_sample(rng, hits)
        hit = _trim_silence(sample.audio)
        level_db = rng.uniform(*SYNTH.random_hit_level_db_range)
        logger.debug(
            "Hit {path} at {time:.2f} s: {level_db:+.1f} dB",
            path=sample.name,
            time=grid.to_drop_seconds(start),
            level_db=level_db,
        )
        gain = _relative_gain(level_db, reference, _compute_level(hit))
        _mix_into(out, hit, start, gain)
    return out


def _render_impact(
    rng: np.random.Generator, impacts: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The impact track (in some drops): one impact at the start of a phrase."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not impacts or rng.random() >= SYNTH.impact_layer_probability:
        return out
    sample = _draw_sample(rng, impacts)
    impact = _trim_silence(sample.audio)
    bar = rng.choice(np.arange(0, grid.bars, SYNTH.bars_per_phrase))
    level_db = rng.uniform(*SYNTH.impact_level_db_range)
    start = round(bar * grid.bar)
    logger.debug(
        "Impact {path} at {time:.2f} s: {level_db:+.1f} dB",
        path=sample.name,
        time=grid.to_drop_seconds(start),
        level_db=level_db,
    )
    gain = _relative_gain(level_db, reference, _compute_level(impact))
    _mix_into(out, impact, start, gain)
    return out


def create_drop(
    bank: Bank, rng: np.random.Generator, held_out: bool = False
) -> LabeledDrop:
    """A random drop of the configured duration and each kick onset in seconds.

    With `held_out`, the kick is from the held-out kicks (for validation drops).
    """
    kick_sample = _draw_kick(rng, bank.kicks, held_out)
    grid = _draw_grid(rng)
    kick_pattern = _draw_kick_pattern(rng, grid.bars)
    kick_starts = grid.to_samples(kick_pattern.positions)
    kick = _trim_silence(kick_sample.audio)
    kick_level = _compute_level(kick)

    mix = (
        _render_kicks(rng, kick, kick_starts, kick_pattern.in_roll, grid)
        + _render_loops(rng, bank.loops, kick_starts, grid, kick_level)
        + _render_claps(rng, bank.claps, grid, kick_level)
        + _render_hits(rng, bank.hits, grid, kick_level)
        + _render_impact(rng, bank.impacts, grid, kick_level)
    )

    drop_length = round(SYNTH.drop_duration_seconds * SAMPLE_RATE)
    drop = master_drop(rng, mix[grid.offset : grid.offset + drop_length])

    audio = librosa.util.normalize(drop) * librosa.db_to_amplitude(
        SYNTH.drop_peak_level_db
    )
    in_drop = (kick_starts >= grid.offset) & (kick_starts < grid.offset + drop_length)
    onsets = librosa.samples_to_time(kick_starts[in_drop] - grid.offset, sr=SAMPLE_RATE)
    return LabeledDrop(
        audio=audio.astype(np.float32),
        onsets=onsets,
        bpm=grid.bpm,
        kick_name=kick_sample.name,
    )
