from dataclasses import dataclass, field

import numpy as np

from src.services.synthesis.bank import SampleBank
from src.services.synthesis.effects import stretch
from src.services.synthesis.mix import Mix
from src.services.synthesis.track import Grid, Track


@dataclass
class KickPatternGenerator:
    grid: Grid
    probability: float = 0.5
    steps_per_beat: int = 4
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def generate(self) -> np.ndarray:
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        if not 0 < self.steps_per_beat:
            raise ValueError("Steps per beat must be positive.")

        step = 1 / self.steps_per_beat
        steps = np.arange(0, self.grid.beats, step)
        return steps[self.rng.random(len(steps)) < self.probability]


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
