from math import floor

import numpy as np

from src.models.audio import Signal
from src.services.synthesis.effects import Effect, cut
from src.services.synthesis.grid import Grid


class Track:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.samples = floor(grid.bars * grid.bar)
        self.signal = np.zeros(self.samples, dtype=np.float32)
        self.effects: list[Effect] = []

    def insert(
        self,
        signal: Signal,
        beat_position: float,
        blend: bool = True,
    ) -> None:
        if beat_position < 0:
            raise ValueError("Beat cannot be negative.")
        start = self.grid.to_sample(beat_position)
        if start >= self.samples:
            return
        cut_signal = cut(signal, max(0, self.samples - start), sr=self.grid.sr)
        end = start + len(cut_signal)
        if blend:
            self.signal[start:end] += cut_signal
        else:
            self.signal[start:end] = cut_signal

    def add_effects(self, *effects: Effect) -> None:
        self.effects.extend(effects)

    def process(self) -> Signal:
        wet_signal = self.signal.copy()
        for effect in self.effects:
            wet_signal = effect.process(wet_signal, self.grid.sr)
        return wet_signal
