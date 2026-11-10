from dataclasses import dataclass, field

import librosa
import numpy as np
from loguru import logger

from src.core.config import KickFxConfig, KickRollConfig, SectionConfig
from src.models.audio import SampleKind
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.draws import draw_count, draw_log_uniform
from src.services.synthesis.effects import (
    Filter,
    Section,
    measure_loudest_rms,
    trim_to_attack,
)
from src.services.synthesis.grid import Grid
from src.services.synthesis.track import Track


@dataclass
class KickPatternGenerator:
    """Draw kicks on a grid. Coarser subdivisions have a higher chance."""

    grid: Grid
    """Grid the pattern is generated for."""
    probability: float = 0.98
    """Kick probability on the beat itself, between 0 and 1."""
    weight: float = 0.12
    """Multiplier applied per subdivision level, between 0 and 1."""
    off_beat_probability: float = 0.0
    """Extra chance of a kick on the 1/8 off-beat of each beat, on any grid,
    between 0 and 1."""
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    """Random number generator used to draw the pattern."""

    def generate(self) -> np.ndarray:
        """Draw kick positions. Raise ValueError for invalid probabilities."""
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1.")
        if not 0 <= self.off_beat_probability <= 1:
            raise ValueError("Off-beat probability must be between 0 and 1.")

        beat_weights = self.probability * self.weight ** self._calculate_depths(
            self.grid.steps_per_beat
        )
        weights = np.tile(beat_weights, self.grid.beats)
        steps = (
            np.arange(self.grid.beats * self.grid.steps_per_beat)
            / self.grid.steps_per_beat
        )
        off_beats = np.arange(self.grid.beats) + 0.5
        return np.union1d(
            steps[self.rng.random(len(steps)) < weights],
            off_beats[
                self.rng.random(len(off_beats)) < self.off_beat_probability
            ],
        )

    @staticmethod
    def _calculate_depths(steps_per_beat: int) -> np.ndarray:
        """Return each step's subdivision depth."""
        if steps_per_beat == 1:
            return np.zeros(1, dtype=int)

        factor = next(
            (
                f
                for f in range(2, int(steps_per_beat**0.5) + 1)
                if steps_per_beat % f == 0
            ),
            steps_per_beat,
        )
        coarse = KickPatternGenerator._calculate_depths(
            steps_per_beat // factor
        )
        depths = np.full(steps_per_beat, coarse.max() + 1)
        depths[::factor] = coarse
        return depths


@dataclass(frozen=True)
class BeatSpan:
    start: int
    """First beat."""
    end: int
    """Beat after the span (exclusive)."""


@dataclass(frozen=True)
class KickTrack:
    track: Track
    beats: np.ndarray
    """Kick positions in beats."""
    level: float
    """Loudest-window RMS used to level other layers."""
    gaps: list[BeatSpan]
    """Spans where kicks stop."""
    rolls: list[BeatSpan]
    """Spans where kicks play on every roll step."""


def _draw_spans(
    grid: Grid,
    config: SectionConfig,
    taken: np.ndarray,
    rng: np.random.Generator,
) -> list[BeatSpan]:
    """Draw spans on unused beats and mark them as taken."""
    spans = []
    for _ in range(draw_count(config.section_count_weights, rng)):
        beats = int(rng.choice(config.section_beats))
        free = [
            start
            for start in range(grid.beats - beats + 1)
            if not taken[start : start + beats].any()
        ]
        if not free:
            continue
        start = free[rng.integers(len(free))]
        taken[start : start + beats] = True
        spans.append(BeatSpan(start=start, end=start + beats))
    return spans


def draw_kick_fx(
    grid: Grid,
    config: KickFxConfig,
    rng: np.random.Generator,
    taken: np.ndarray | None = None,
) -> list[Section]:
    """Draw non-overlapping high-pass sections on unused beats."""
    if taken is None:
        taken = np.zeros(grid.beats, dtype=bool)
    sections = []
    for span in _draw_spans(grid, config, taken, rng):
        cutoff = draw_log_uniform(config.high_pass_hz_range, rng)
        q = draw_log_uniform(config.resonance_q_range, rng)
        order = int(rng.choice(config.filter_orders))
        logger.debug(
            f"kick highpass at {cutoff:.0f} Hz, {6 * order} dB/oct, Q {q:.1f}, "
            f"on beats {span.start}-{span.end}"
        )
        sections.append(
            Section(
                Filter(cutoff=cutoff, btype="highpass", order=order, q=q),
                start=grid.to_sample(span.start),
                end=grid.to_sample(span.end),
            )
        )
    return sections


@dataclass(frozen=True)
class KickPattern:
    beats: np.ndarray
    """Beat positions of the kicks, in order."""
    gains: np.ndarray
    """Gain of each kick."""


def _add_rolls(
    beats: np.ndarray,
    rolls: list[BeatSpan],
    config: KickRollConfig,
    rng: np.random.Generator,
) -> KickPattern:
    """Replace each roll span with kicks at one rate and level."""
    gains = np.ones(len(beats))
    for roll in rolls:
        steps = int(rng.choice(config.steps_per_beat))
        db = rng.uniform(*config.level_db_range)
        logger.debug(
            f"kick roll of 1/{4 * steps} notes at {db:.1f} dB "
            f"on beats {roll.start}-{roll.end}"
        )
        outside = (beats < roll.start) | (beats >= roll.end)
        roll_beats = (
            roll.start + np.arange((roll.end - roll.start) * steps) / steps
        )
        beats = np.concatenate([beats[outside], roll_beats])
        gains = np.concatenate(
            [
                gains[outside],
                np.full(len(roll_beats), librosa.db_to_amplitude(db)),
            ]
        )
    order = np.argsort(beats)
    return KickPattern(beats=beats[order], gains=gains[order])


def draw_kick_track(
    grid: Grid,
    bank: SampleBank,
    gap_config: SectionConfig,
    roll_config: KickRollConfig,
    fx_config: KickFxConfig,
    off_beat_range: tuple[float, float],
    rng: np.random.Generator,
) -> KickTrack:
    """Draw kicks with gaps, rolls, and effects.

    Draw the extra off-beat chance once per drop. Each kick cuts the prior
    tail. The tail continues into gaps, but gaps, rolls, and effects do not overlap.
    """
    kick = trim_to_attack(bank.draw(SampleKind.KICK, rng).signal)
    off_beat = rng.uniform(*off_beat_range)
    logger.debug(f"kick off-beat chance {off_beat:.2f}")
    beats = KickPatternGenerator(
        grid, off_beat_probability=off_beat, rng=rng
    ).generate()
    taken = np.zeros(grid.beats, dtype=bool)
    gaps = _draw_spans(grid, gap_config, taken, rng)
    for gap in gaps:
        logger.debug(f"kick gap on beats {gap.start}-{gap.end}")
        beats = beats[(beats < gap.start) | (beats >= gap.end)]
    rolls = _draw_spans(grid, roll_config, taken, rng)
    pattern = _add_rolls(beats, rolls, roll_config, rng)
    track = Track(grid)
    for beat, gain in zip(pattern.beats, pattern.gains, strict=True):
        track.insert(kick * gain, beat, blend=False)
    track.add_effects(*draw_kick_fx(grid, fx_config, rng, taken))
    return KickTrack(
        track=track,
        beats=pattern.beats,
        level=measure_loudest_rms(kick, grid.sr),
        gaps=gaps,
        rolls=rolls,
    )
