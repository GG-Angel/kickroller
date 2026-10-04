"""Synthetic rawstyle drops with exact kick onset labels.

A drop is rendered on a bar grid that starts up to one bar before the drop:

- Kicks: beats may be empty, have one kick, or contain a kickroll. Each drop
    uses either straight or triplet subdivisions, and each kick cuts the tail of
    the kick before it.
- Loops: stretched to the drop tempo, high-passed and ducked at each kick.
- Claps on beats 2 and 4, random hits on the 1/16 grid, and sometimes an impact.

The levels are relative to the kick. Mastering (EQ, clipping, limiting and
encoding) is not done here, because it is a training augmentation.
"""

from dataclasses import dataclass
from math import ceil

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from core.settings import SETTINGS

from .models import Bank, LabeledDrop, Sample

SAMPLE_RATE = SETTINGS.sample_rate
SYNTH = SETTINGS.synth


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


@dataclass(frozen=True)
class KickPattern:
    positions: tuple[float, ...]
    in_roll: tuple[bool, ...]


def _compute_rms(signal: np.ndarray) -> float:
    """Compute the RMS level of a signal."""
    return float(np.sqrt(np.mean(np.square(signal))))


def _compute_attack_rms(sound: np.ndarray) -> float:
    """The RMS level of the attack and body of a one-shot."""
    attack_length = round(SYNTH.one_shot_attack_window_seconds * SAMPLE_RATE)
    return _compute_rms(sound[:attack_length])


def _relative_gain(level_db: float, reference: float, level: float) -> float:
    """The gain that puts a sound with RMS `level` at `level_db` relative to `reference`."""
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


def _fit_loop_to_grid(
    rng: np.random.Generator, sample: Sample, grid: Grid
) -> np.ndarray:
    """A loop stretched to the drop tempo, high-passed and repeated over the grid."""
    loop = sample.audio
    if sample.bpm != grid.bpm:
        loop = librosa.effects.time_stretch(loop, rate=grid.bpm / sample.bpm)
    bars = max(1, round(len(loop) / grid.bar))
    loop = librosa.util.fix_length(loop, size=round(bars * grid.bar))
    loop = _high_pass(loop, rng.uniform(*SYNTH.loop_high_pass_cutoff_hz_range))
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


def _draw_sample(rng: np.random.Generator, samples: list[Sample]) -> Sample:
    """Draw a random sample from the list of samples."""
    return samples[int(rng.integers(len(samples)))]


def _draw_one_shot(rng: np.random.Generator, samples: list[Sample]) -> np.ndarray:
    """Draw a random one-shot sample and trim its silence."""
    return _trim_silence(_draw_sample(rng, samples).audio)


def _draw_grid(rng: np.random.Generator) -> Grid:
    """Draw a random grid with a BPM chosen according to the configured weights."""
    bpm = int(
        rng.choice(
            [bpm for bpm, _ in SYNTH.bpm_probability_weights],
            p=[weight for _, weight in SYNTH.bpm_probability_weights],
        )
    )
    return Grid(bpm=bpm, phase=rng.random())


def _draw_kick_pattern_for_beat(
    rng: np.random.Generator,
    beat: int,
    kickless_beat_rate: float,
    kickroll_chance: float,
    triplet: bool,
) -> KickPattern:
    """Draw the kick pattern for a single beat (no kick, on-beat kick, or kickroll)."""
    if rng.random() < kickless_beat_rate:
        return KickPattern(positions=(), in_roll=())  # no kick
    if rng.random() >= kickroll_chance:
        return KickPattern(positions=(float(beat),), in_roll=(False,))  # on-beat kick

    # kickroll
    subdivisions = (1, 3, 6) if triplet else (1, 2, 4)
    weights = np.asarray(SYNTH.kickroll_hits_per_beat_weights)
    count = int(rng.choice(subdivisions, p=weights / weights.sum()))
    positions = beat + np.arange(count) / count
    selected = rng.random(count) < 0.5
    if not np.any(selected):
        selected[rng.integers(count)] = True

    kicks = tuple(float(position) for position in positions[selected])
    return KickPattern(positions=kicks, in_roll=(True,) * len(kicks))


def _draw_kick_pattern(rng: np.random.Generator, bars: int) -> KickPattern:
    """Draw the kick pattern for the entire drop."""
    kickless_drop_probability = rng.uniform(*SYNTH.kickless_drop_probability_range)
    if rng.random() < kickless_drop_probability:
        return KickPattern(positions=(), in_roll=())  # no kicks for this drop

    kickless_beat_probability = rng.uniform(*SYNTH.kickless_beat_probability_range)
    kickroll_probability = rng.uniform(*SYNTH.per_beat_kickroll_probability_range)
    triplet = rng.random() < SYNTH.triplet_grid_probability
    positions: list[float] = []
    in_roll: list[bool] = []
    for bar in range(bars):
        for beat in range(bar * SYNTH.beats_per_bar, (bar + 1) * SYNTH.beats_per_bar):
            pattern = _draw_kick_pattern_for_beat(
                rng, beat, kickless_beat_probability, kickroll_probability, triplet
            )
            positions.extend(pattern.positions)
            in_roll.extend(pattern.in_roll)
    return KickPattern(positions=tuple(positions), in_roll=tuple(in_roll))


def _render_kicks(
    rng: np.random.Generator,
    kick: np.ndarray,
    starts: np.ndarray,
    in_roll: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    """The kick track: each kick plays until the next kick starts."""
    out = np.zeros(grid.length, dtype=np.float32)
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
    """The loop track: up to the configured number of loops, ducked at each kick."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not loops:
        return out
    for _ in range(int(rng.integers(0, SYNTH.max_simultaneous_loops + 1))):
        loop = _fit_loop_to_grid(rng, _draw_sample(rng, loops), grid)
        level_db = rng.uniform(*SYNTH.loop_level_db_range)
        out += _relative_gain(level_db, reference, _compute_rms(loop)) * loop
    return out * _make_sidechain_curve(rng, starts, grid.length)


def _render_claps(
    rng: np.random.Generator, claps: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The clap track (in some drops): one clap sample on beats 2 and 4."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not claps or rng.random() >= SYNTH.clap_layer_probability:
        return out
    clap = _draw_one_shot(rng, claps)
    level_db = rng.uniform(*SYNTH.clap_level_db_range)
    gain = _relative_gain(level_db, reference, _compute_attack_rms(clap))
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
        hit = _draw_one_shot(rng, hits)
        level_db = rng.uniform(*SYNTH.random_hit_level_db_range)
        gain = _relative_gain(level_db, reference, _compute_attack_rms(hit))
        _mix_into(out, hit, start, gain)
    return out


def _render_impact(
    rng: np.random.Generator, impacts: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The impact track (in some drops): one impact at the start of a phrase."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not impacts or rng.random() >= SYNTH.impact_layer_probability:
        return out
    impact = _draw_one_shot(rng, impacts)
    bar = rng.choice(np.arange(0, grid.bars, SYNTH.bars_per_phrase))
    level_db = rng.uniform(*SYNTH.impact_level_db_range)
    gain = _relative_gain(level_db, reference, _compute_attack_rms(impact))
    _mix_into(out, impact, round(bar * grid.bar), gain)
    return out


def create_drop(bank: Bank, rng: np.random.Generator) -> LabeledDrop:
    """A random drop of the configured duration and each kick onset in seconds."""
    if not bank.kicks:
        raise ValueError("The bank has no kicks")

    grid = _draw_grid(rng)
    kick_pattern = _draw_kick_pattern(rng, grid.bars)
    kick_starts = grid.to_samples(beats=np.asarray(kick_pattern.positions, dtype=float))
    in_roll = np.asarray(kick_pattern.in_roll, dtype=bool)

    kick = _draw_one_shot(rng, bank.kicks)
    kick_attack_rms = _compute_attack_rms(kick)

    mix = (
        _render_kicks(rng, kick, kick_starts, in_roll, grid)
        + _render_loops(rng, bank.loops, kick_starts, grid, kick_attack_rms)
        + _render_claps(rng, bank.claps, grid, kick_attack_rms)
        + _render_hits(rng, bank.hits, grid, kick_attack_rms)
        + _render_impact(rng, bank.impacts, grid, kick_attack_rms)
    )

    drop_length = round(SYNTH.drop_duration_seconds * SAMPLE_RATE)
    drop = mix[grid.offset : grid.offset + drop_length]

    audio = librosa.util.normalize(drop) * librosa.db_to_amplitude(
        SYNTH.drop_peak_level_db
    )
    in_drop = (kick_starts >= grid.offset) & (kick_starts < grid.offset + drop_length)
    onsets = librosa.samples_to_time(kick_starts[in_drop] - grid.offset, sr=SAMPLE_RATE)
    return LabeledDrop(audio=audio.astype(np.float32), onsets=onsets)
