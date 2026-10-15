from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.core.config import KickFxConfig
from src.models.audio import SampleKind
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.draws import draw_count
from src.services.synthesis.effects import (
    Filter,
    Section,
    loudest_rms,
    trim_signal,
)
from src.services.synthesis.grid import Grid
from src.services.synthesis.patterns.pattern import PatternGenerator
from src.services.synthesis.track import Track


@dataclass
class KickPatternGenerator(PatternGenerator):
    """Randomly place kicks on a grid, favoring steps on coarser subdivisions.

    Each step in a beat fires with probability ``probability * weight ** depth``,
    where ``depth`` is how many subdivision levels the step sits below the beat
    (e.g. ``steps_per_beat=4`` gives ``p, p·w², p·w, p·w²``).
    """

    grid: Grid
    """Grid the pattern is generated for."""
    probability: float = 0.98
    """Kick probability on the beat itself, between 0 and 1."""
    weight: float = 0.12
    """Multiplier applied per subdivision level, between 0 and 1."""
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    """Random number generator used to draw the pattern."""

    def generate(self) -> np.ndarray:
        """Return the beat positions of the kicks drawn across the whole grid.

        Raises:
            ValueError: If ``probability`` or ``weight`` is outside [0, 1], or
                ``steps_per_beat`` is not positive.
        """
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1.")

        beat_weights = self.probability * self.weight ** self._calculate_depths(
            self.grid.steps_per_beat
        )
        weights = np.tile(beat_weights, self.grid.beats)
        steps = (
            np.arange(self.grid.beats * self.grid.steps_per_beat)
            / self.grid.steps_per_beat
        )
        return steps[self.rng.random(len(steps)) < weights]

    @staticmethod
    def _calculate_depths(steps_per_beat: int) -> np.ndarray:
        """Return the subdivision depth of each step within a beat.

        The beat is split by its largest prime factors first and by smaller ones
        at finer levels, e.g. ``4 -> [0, 2, 1, 2]`` and ``6 -> [0, 2, 1, 2, 1, 2]``.
        """
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
class KickTrack:
    track: Track
    beats: np.ndarray
    """Beat positions of the kicks."""
    level: float
    """``loudest_rms`` of the kick, which the other layers are leveled to."""


def draw_kick_fx(
    grid: Grid, config: KickFxConfig, rng: np.random.Generator
) -> list[Section]:
    """Sections of whole beats that go through a high-pass or low-pass.

    The sections start on any beat and do not overlap.
    """
    taken = np.zeros(grid.beats, dtype=bool)
    sections = []
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
        btype = str(rng.choice(["highpass", "lowpass"]))
        cutoff = _draw_log_uniform(
            config.high_pass_hz_range
            if btype == "highpass"
            else config.low_pass_hz_range,
            rng,
        )
        logger.debug(
            f"kick {btype} at {cutoff:.0f} Hz on beats {start}-{start + beats}"
        )
        sections.append(
            Section(
                Filter(cutoff=cutoff, btype=btype),
                start=grid.to_sample(start),
                end=grid.to_sample(start + beats),
            )
        )
    return sections


def _draw_log_uniform(
    value_range: tuple[float, float], rng: np.random.Generator
) -> float:
    low, high = np.log(value_range)
    return float(np.exp(rng.uniform(low, high)))


def draw_kick_track(
    grid: Grid,
    bank: SampleBank,
    fx_config: KickFxConfig,
    rng: np.random.Generator,
) -> KickTrack:
    """One kick at the positions of a random kick pattern, with kick FX.

    Each kick cuts the tail of the one before it.
    """
    kick = trim_signal(bank.draw(SampleKind.KICK, rng).signal)
    beats = KickPatternGenerator(grid, rng=rng).generate()
    track = Track(grid)
    for beat in beats:
        track.insert(kick, beat, blend=False)
    track.add_effects(*draw_kick_fx(grid, fx_config, rng))
    return KickTrack(track=track, beats=beats, level=loudest_rms(kick))
