import librosa
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    bpm: int
    sr: int
    beats_per_bar: int

    @property
    def beat(self) -> float:
        """The length of a beat in samples."""
        return 60.0 * self.sr / self.bpm

    @property
    def bar(self) -> float:
        """The length of a bar in samples."""
        return self.beats_per_bar * self.beat

    def convert_beats_to_samples(self, beats: np.ndarray) -> np.ndarray:
        """Positions in beats as positions in samples."""
        times = beats * 60.0 / self.bpm
        return librosa.time_to_samples(times, sr=self.sr)


class Synthesizer:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
