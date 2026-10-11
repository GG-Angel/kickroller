from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.core.config import CONFIG, SynthConfig
from src.services.synthesis.bank import Sample, SampleBank, SampleKind
from src.services.synthesis.mix import Mix
from src.services.synthesis.track import Grid, Track


class PatternGenerator(Protocol):
    def generate(self) -> np.ndarray: ...


@dataclass
class KickPatternGenerator(PatternGenerator):
    """Randomly place kicks on a grid, favoring steps on coarser subdivisions.

    Each step in a beat fires with probability ``probability * weight ** depth``,
    where ``depth`` is how many subdivision levels the step sits below the beat
    (e.g. ``steps_per_beat=4`` gives ``p, p·w², p·w, p·w²``).
    """

    grid: Grid
    """Grid the pattern is generated for."""
    probability: float = 1.0
    """Kick probability on the beat itself, between 0 and 1."""
    weight: float = 0.3
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


@dataclass
class MelodyPatternGenerator(PatternGenerator):
    grid: Grid
    sample: Sample
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def generate(self) -> np.ndarray:
        return np.array([0, 16])


def draw_grid(config: SynthConfig, rng: np.random.Generator) -> Grid:
    bpm = int(rng.choice(config.bpms))
    return Grid(bpm=bpm, bars=4)


def generate_drop(
    bank: SampleBank,
    config: SynthConfig = CONFIG.synth,
) -> np.ndarray:
    rng = np.random.default_rng()
    grid = draw_grid(config, rng)

    kick_track = Track(grid)
    kick = bank.draw(SampleKind.KICK)
    kick_pattern = KickPatternGenerator(grid, rng=rng).generate()
    for beat in kick_pattern:
        kick_track.insert(kick.signal, beat, blend=False)

    melody_track = Track(grid)
    melody = bank.draw(SampleKind.MELODY, loops_only=True).stretch(melody_track)
    for beat in MelodyPatternGenerator(grid, melody, rng=rng).generate():
        melody_track.insert(melody.signal, beat, blend=False)

    melody_track.sidechain(grid.to_samples(kick_pattern), ratio=0.5)

    mix = Mix()
    mix.add(kick_track, melody_track)
    return mix.mix()
