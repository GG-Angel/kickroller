from dataclasses import dataclass

import librosa
import numpy as np

from src.core.config import CONFIG


@dataclass(frozen=True)
class Grid:
    bpm: int
    bars: int
    beats_per_bar: int = 4
    sr: int = CONFIG.sr

    @property
    def beat(self) -> float:
        """Return the duration of a single beat in samples."""
        return 60.0 * self.sr / self.bpm

    @property
    def bar(self) -> float:
        """Return the duration of a single bar in samples."""
        return self.beat * self.beats_per_bar

    def to_seconds(self, beats: np.ndarray) -> np.ndarray:
        """Convert an array of beats to an array of time in seconds."""
        return beats * 60.0 / self.bpm

    def to_samples(self, beats: np.ndarray) -> np.ndarray:
        """Convert an array of beats to an array of sample indices."""
        return librosa.time_to_samples(self.to_seconds(beats), sr=self.sr)

    def to_sample(self, beat: float) -> int:
        """Convert a single beat to a sample index."""
        return self.to_samples(np.array([beat]))[0]
