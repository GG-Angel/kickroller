from dataclasses import dataclass
from math import floor

import librosa
import numpy as np

from src.core.config import CONFIG
from src.models.audio import Signal
from src.services.synthesis.bank import SampleBank


@dataclass(frozen=True)
class Grid:
    bpm: int
    sr: int = CONFIG.sr
    beats_per_bar: int = 4

    @property
    def beat(self) -> float:
        """Return the duration of a single beat in samples."""
        return 60.0 * self.sr / self.bpm

    @property
    def bar(self) -> float:
        """Return the duration of a single bar in samples."""
        return self.beat * self.beats_per_bar

    def to_samples(self, beats: np.ndarray) -> np.ndarray:
        """Convert an array of beats to an array of sample indices."""
        times = beats * 60.0 / self.bpm
        return librosa.time_to_samples(times, sr=self.sr)

    def to_sample(self, beat: float) -> int:
        """Convert a single beat to a sample index."""
        return self.to_samples(np.array([beat]))[0]


class Track:
    def __init__(self, grid: Grid, bars: int) -> None:
        self.grid = grid
        self.samples = floor(bars * grid.bar)
        self.signal = np.zeros(self.samples)

    def insert(self, signal: Signal, beat: float) -> None:
        start = self.grid.to_sample(beat)
        bounded_signal = signal[: max(0, self.samples - start)]
        self.signal[start : start + len(bounded_signal)] = bounded_signal


def draw_grid() -> Grid:
    return Grid(bpm=160)


def generate_kick_pattern() -> np.ndarray:
    return np.array([0, 1, 2, 3, 3.5, 4, 5, 6, 6.5, 7, 8, 8.5])


def generate_drop(bank: SampleBank) -> np.ndarray:
    grid = draw_grid()

    kick = bank.draw_kick()
    kick_track = Track(grid=grid, bars=4)

    for beat in generate_kick_pattern():
        kick_track.insert(kick.signal, beat)

    return kick_track.signal
