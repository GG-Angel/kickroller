from math import floor

import numpy as np
from src.models.audio import Signal
from src.services.synthesis.daw.effects import Cut, Effect
from src.services.synthesis.daw.grid import Grid


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
        clip = Cut(max(0, self.samples - start)).process(signal, self.grid.sr)
        end = start + len(clip)
        if blend:
            self.signal[start:end] += clip
        else:
            self.signal[start:end] = clip

    def add_effects(self, *effects: Effect) -> None:
        self.effects.extend(effects)

    def process(self) -> Signal:
        wet_signal = self.signal.copy()
        for effect in self.effects:
            wet_signal = effect.process(wet_signal, self.grid.sr)
        return wet_signal
