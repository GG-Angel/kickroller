"""Synthetic rawstyle drops with exact kick onset labels.

A drop is rendered on a bar grid that starts up to one bar before the drop:

- Kicks: one kick sample per drop, on the beats (some are missing), on 1/8
  off-beats, on some syncopated 1/16 and triplet positions, and in rolls at the
  end of some bars. Some bars have no kicks. Each kick cuts the tail of the
  kick before it.
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
DROP_SECONDS = 12.0
DROP_LENGTH = round(DROP_SECONDS * SAMPLE_RATE)
BEATS_PER_BAR = 4
SIXTEENTHS_PER_BEAT = 4
PHRASE_BARS = 4
PEAK_DB = -1.0  # the peak level of the drop

# Tempo: TARGET_BPM in TARGET_BPM_CHANCE of the drops, else a whole BPM in BPM_RANGE.
TARGET_BPM = 160
TARGET_BPM_CHANCE = 0.5
BPM_RANGE = (150, 170)

# Kicks. Positions are in beats.
KICKLESS_BAR_CHANCE = 0.05
MISSING_RATE = (0.0, 0.15)  # the chance of no kick on a beat (one rate per drop)
OFF_BEAT_RATE = (0.0, 0.3)  # the chance of a 1/8 off-beat kick (one rate per drop)
SYNCOPATED_CHANCE = 0.03  # the chance of a kick on one of SYNCOPATED_POSITIONS
SYNCOPATED_POSITIONS = (1 / 4, 1 / 3, 2 / 3, 3 / 4)  # in a beat
ROLL_CHANCE = 0.2  # per bar
ROLL_BEATS = (1, 2, 4)  # a roll is on the last beats of a bar
ROLL_STEPS = (1 / 4, 1 / 3, 1 / 2)  # 1/16 notes, triplets and 1/8 notes
KICK_LEVEL_DB = (-1.0, 0.0)
ROLL_LEVEL_DB = (-6.0, 0.0)
CUT_FADE = round(0.003 * SAMPLE_RATE)  # the fade-out when the next kick cuts a tail

# The other layers. Levels are in dB relative to the kick.
MAX_LOOPS = 3
LOOP_LEVEL_DB = (-18.0, 3.0)
HIGH_PASS_HZ = (100.0, 250.0)  # the kick owns the sub band
HIGH_PASS_ORDER = 2
DUCK_DEPTH = (0.4, 1.0)  # 1 is silence at the kick
DUCK_RELEASE_SECONDS = (0.05, 0.25)  # the time constant of the recovery
CLAP_CHANCE = 0.6
CLAP_LEVEL_DB = (-12.0, -2.0)
MEAN_HITS = 8.0  # per drop
HIT_LEVEL_DB = (-18.0, -3.0)
IMPACT_CHANCE = 0.3
IMPACT_LEVEL_DB = (-12.0, 0.0)

# One-shots start where they are louder than TRIM_TOP_DB below their peak.
TRIM_TOP_DB = 40.0
TRIM_FRAME_LENGTH = 64
TRIM_HOP_LENGTH = 16
ATTACK_LENGTH = round(0.1 * SAMPLE_RATE)  # the part of a one-shot that sets its level


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
        return BEATS_PER_BAR * self.beat

    @property
    def offset(self) -> int:
        """The offset of the drop in samples from the first bar."""
        return int(self.phase * self.bar)

    @property
    def bars(self) -> int:
        """The number of bars that cover the drop including the offset."""
        return ceil((self.offset + DROP_LENGTH) / self.bar)

    @property
    def length(self) -> int:
        """The length of the drop in samples including the offset."""
        return ceil(self.bars * self.bar)

    def to_samples(self, beats: np.ndarray) -> np.ndarray:
        """Positions in beats as positions in samples."""
        return librosa.time_to_samples(beats * 60.0 / self.bpm, sr=SAMPLE_RATE)


# Levels and mixing


def _compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def _compute_attack_rms(sound: np.ndarray) -> float:
    """The RMS level of the attack and body of a one-shot."""
    return _compute_rms(sound[:ATTACK_LENGTH])


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
    fade = min(CUT_FADE, length)
    cut = sound[:length].copy()
    cut[length - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return cut


def _high_pass(signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """High-pass filter the signal at `cutoff_hz`."""
    sos = butter(
        HIGH_PASS_ORDER, cutoff_hz, btype="highpass", fs=SAMPLE_RATE, output="sos"
    )
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


# Samples


def _draw_sample(rng: np.random.Generator, samples: list[Sample]) -> Sample:
    return samples[int(rng.integers(len(samples)))]


def _trim_silence(sound: np.ndarray) -> np.ndarray:
    """A one-shot without silence at its ends, so that it starts at its attack."""
    trimmed, _ = librosa.effects.trim(
        sound,
        top_db=TRIM_TOP_DB,
        frame_length=TRIM_FRAME_LENGTH,
        hop_length=TRIM_HOP_LENGTH,
    )
    return trimmed


def _draw_one_shot(rng: np.random.Generator, samples: list[Sample]) -> np.ndarray:
    return _trim_silence(_draw_sample(rng, samples).audio)


# Grid and kick pattern


def _draw_grid(rng: np.random.Generator) -> Grid:
    if rng.random() < TARGET_BPM_CHANCE:
        bpm = TARGET_BPM
    else:
        bpm = int(rng.integers(low=BPM_RANGE[0], high=BPM_RANGE[1] + 1))
    return Grid(bpm=bpm, phase=rng.random())


def _draw_beat(
    rng: np.random.Generator, beat: int, missing_rate: float, off_beat_rate: float
) -> list[float]:
    """The kicks of one beat: on the beat, then a 1/8 off-beat or a syncopated kick."""
    kicks: list[float] = []
    if rng.random() >= missing_rate:
        kicks.append(beat)
    if rng.random() < off_beat_rate:
        kicks.append(beat + 1 / 2)
    elif rng.random() < SYNCOPATED_CHANCE:
        kicks.append(beat + float(rng.choice(SYNCOPATED_POSITIONS)))
    return kicks


def _draw_roll(rng: np.random.Generator, start: int, beats: int) -> list[float]:
    """The kicks of a roll of `beats` beats from beat `start`, at one of ROLL_STEPS."""
    step = float(rng.choice(ROLL_STEPS))
    return list(start + step * np.arange(round(beats / step)))


def _draw_kick_pattern(
    rng: np.random.Generator, bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Kick positions in beats from the first bar, and whether each kick is in a roll."""
    missing_rate = rng.uniform(*MISSING_RATE)
    off_beat_rate = rng.uniform(*OFF_BEAT_RATE)
    beats: list[float] = []
    in_roll: list[bool] = []
    for bar in range(bars):
        if rng.random() < KICKLESS_BAR_CHANCE:
            continue
        roll_beats = int(rng.choice(ROLL_BEATS)) if rng.random() < ROLL_CHANCE else 0
        roll_start = (bar + 1) * BEATS_PER_BAR - roll_beats
        for beat in range(bar * BEATS_PER_BAR, roll_start):
            kicks = _draw_beat(rng, beat, missing_rate, off_beat_rate)
            beats += kicks
            in_roll += [False] * len(kicks)
        if roll_beats:
            roll = _draw_roll(rng, roll_start, roll_beats)
            beats += roll
            in_roll += [True] * len(roll)
    return np.array(beats, dtype=float), np.array(in_roll, dtype=bool)


# Tracks


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
        rng.uniform(*ROLL_LEVEL_DB, size=len(starts)),
        rng.uniform(*KICK_LEVEL_DB, size=len(starts)),
    )
    ends = np.append(starts[1:], grid.length)
    for start, end, level_db in zip(starts, ends, levels_db, strict=True):
        gain = float(librosa.db_to_amplitude(level_db))
        _mix_into(out, _cut_tail(kick, end - start), start, gain)
    return out


def _fit_loop_to_grid(
    rng: np.random.Generator, sample: Sample, grid: Grid
) -> np.ndarray:
    """A loop stretched to the drop tempo, high-passed and repeated over the grid.

    The loop is padded or cut to whole bars, and it starts on the first bar.
    """
    loop = sample.audio
    if sample.bpm != grid.bpm:
        loop = librosa.effects.time_stretch(loop, rate=grid.bpm / sample.bpm)
    bars = max(1, round(len(loop) / grid.bar))
    loop = librosa.util.fix_length(loop, size=round(bars * grid.bar))
    loop = _high_pass(loop, rng.uniform(*HIGH_PASS_HZ))
    return np.resize(loop, grid.length)


def _make_sidechain_curve(
    rng: np.random.Generator, starts: np.ndarray, length: int
) -> np.ndarray:
    """A gain curve that dips at each kick start and recovers exponentially."""
    if not len(starts):
        return np.ones(length, dtype=np.float32)
    depth = rng.uniform(*DUCK_DEPTH)
    release = rng.uniform(*DUCK_RELEASE_SECONDS) * SAMPLE_RATE
    positions = np.arange(length)
    previous = np.searchsorted(starts, positions, side="right") - 1
    elapsed = positions - starts[np.maximum(previous, 0)]
    curve = 1.0 - depth * np.exp(-elapsed / release)
    return np.where(previous >= 0, curve, 1.0).astype(np.float32)


def _render_loops(
    rng: np.random.Generator,
    loops: list[Sample],
    starts: np.ndarray,
    grid: Grid,
    reference: float,
) -> np.ndarray:
    """The loop track: up to MAX_LOOPS loops, ducked together at each kick."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not loops:
        return out
    for _ in range(int(rng.integers(0, MAX_LOOPS + 1))):
        loop = _fit_loop_to_grid(rng, _draw_sample(rng, loops), grid)
        level_db = rng.uniform(*LOOP_LEVEL_DB)
        out += _relative_gain(level_db, reference, _compute_rms(loop)) * loop
    return out * _make_sidechain_curve(rng, starts, grid.length)


def _render_claps(
    rng: np.random.Generator, claps: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The clap track (in some drops): one clap sample on beats 2 and 4."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not claps or rng.random() >= CLAP_CHANCE:
        return out
    clap = _draw_one_shot(rng, claps)
    level_db = rng.uniform(*CLAP_LEVEL_DB)
    gain = _relative_gain(level_db, reference, _compute_attack_rms(clap))
    for start in grid.to_samples(np.arange(1, grid.bars * BEATS_PER_BAR, 2)):
        _mix_into(out, clap, start, gain)
    return out


def _render_hits(
    rng: np.random.Generator, hits: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The hit track: random one-shots on random 1/16 notes."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not hits:
        return out
    sixteenths = grid.bars * BEATS_PER_BAR * SIXTEENTHS_PER_BEAT
    positions = rng.integers(0, sixteenths, size=rng.poisson(MEAN_HITS))
    for start in grid.to_samples(positions / SIXTEENTHS_PER_BEAT):
        hit = _draw_one_shot(rng, hits)
        level_db = rng.uniform(*HIT_LEVEL_DB)
        gain = _relative_gain(level_db, reference, _compute_attack_rms(hit))
        _mix_into(out, hit, start, gain)
    return out


def _render_impact(
    rng: np.random.Generator, impacts: list[Sample], grid: Grid, reference: float
) -> np.ndarray:
    """The impact track (in some drops): one impact at the start of a phrase."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not impacts or rng.random() >= IMPACT_CHANCE:
        return out
    impact = _draw_one_shot(rng, impacts)
    bar = rng.choice(np.arange(0, grid.bars, PHRASE_BARS))
    level_db = rng.uniform(*IMPACT_LEVEL_DB)
    gain = _relative_gain(level_db, reference, _compute_attack_rms(impact))
    _mix_into(out, impact, round(bar * grid.bar), gain)
    return out


def create_drop(bank: Bank, rng: np.random.Generator) -> LabeledDrop:
    """A random drop of DROP_SECONDS and the onset (in seconds) of each kick in it."""
    if not bank.kicks:
        raise ValueError("The bank has no kicks")
    grid = _draw_grid(rng)
    beats, in_roll = _draw_kick_pattern(rng, grid.bars)
    starts = grid.to_samples(beats)
    kick = _draw_one_shot(rng, bank.kicks)
    reference = _compute_attack_rms(kick)

    mix = (
        _render_kicks(rng, kick, starts, in_roll, grid)
        + _render_loops(rng, bank.loops, starts, grid, reference)
        + _render_claps(rng, bank.claps, grid, reference)
        + _render_hits(rng, bank.hits, grid, reference)
        + _render_impact(rng, bank.impacts, grid, reference)
    )
    drop = mix[grid.offset : grid.offset + DROP_LENGTH]
    audio = librosa.util.normalize(drop) * librosa.db_to_amplitude(PEAK_DB)
    in_drop = (starts >= grid.offset) & (starts < grid.offset + DROP_LENGTH)
    onsets = librosa.samples_to_time(starts[in_drop] - grid.offset, sr=SAMPLE_RATE)
    return LabeledDrop(audio=audio.astype(np.float32), onsets=onsets)
