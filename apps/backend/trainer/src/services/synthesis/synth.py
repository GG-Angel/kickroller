from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.services.synthesis.bank import SampleBank
from src.services.synthesis.effects import stretch
from src.services.synthesis.mix import Mix
from src.services.synthesis.track import Grid, Track


class PatternGenerator(Protocol):
    def generate(self) -> np.ndarray: ...


@dataclass
class KickPatternGenerator(PatternGenerator):
    grid: Grid
    probability: float = 1.0
    weight: float = 0.3
    steps_per_beat: int = 4
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def generate(self) -> np.ndarray:
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1.")
        if not 0 < self.steps_per_beat:
            raise ValueError("Steps per beat must be positive.")

        beat_weights = self.probability * self.weight ** self._calculate_depths(
            self.steps_per_beat
        )
        weights = np.tile(beat_weights, self.grid.beats)
        steps = (
            np.arange(self.grid.beats * self.steps_per_beat)
            / self.steps_per_beat
        )
        return steps[self.rng.random(len(steps)) < weights]

    @staticmethod
    def _calculate_depths(steps_per_beat: int) -> np.ndarray:
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


def draw_grid() -> Grid:
    return Grid(bpm=160, bars=4)


def generate_melodic_pattern() -> np.ndarray:
    return np.array([0, 4])


def generate_drop(bank: SampleBank) -> np.ndarray:
    grid = draw_grid()

    kick_track = Track(grid=grid)
    kick = bank.draw_kick()
    kick_pattern = KickPatternGenerator(grid=grid).generate()
    for beat in kick_pattern:
        kick_track.insert(kick.signal, beat, blend=False)

    melody_track = Track(grid=grid)
    melody = stretch(bank.draw_melodic_loop(), melody_track)
    for beat in generate_melodic_pattern():
        melody_track.insert(melody.signal, beat, blend=False)

    melody_track.sidechain(grid.to_samples(kick_pattern), ratio=1.0)

    mix = Mix()
    mix.add(kick_track, melody_track)
    return mix.mix()
