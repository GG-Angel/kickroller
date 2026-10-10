import numpy as np

from src.services.synthesis.bank import SampleBank
from src.services.synthesis.effects import stretch
from src.services.synthesis.track import Grid, Mix, Track


def draw_grid() -> Grid:
    return Grid(bpm=160, bars=4)


def generate_kick_pattern() -> np.ndarray:
    return np.array([0, 1, 2, 3, 3.5, 4, 5, 6, 6.5, 7, 7.5, 8])


def generate_melodic_pattern() -> np.ndarray:
    return np.array([0, 4])


def generate_drop(bank: SampleBank) -> np.ndarray:
    grid = draw_grid()

    kick_track = Track(grid=grid)
    kick = bank.draw_kick()
    kick_pattern = generate_kick_pattern()
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
