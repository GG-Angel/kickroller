from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.core.config import LayerConfig
from src.services.synthesis.bank import Sample, SampleBank
from src.services.synthesis.grid import Grid
from src.services.synthesis.patterns.pattern import PatternGenerator
from src.services.synthesis.track import Track


@dataclass
class LoopPatternGenerator(PatternGenerator):
    grid: Grid
    sample: Sample
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def generate(self) -> np.ndarray:
        if not self.sample.is_loop:
            raise ValueError(f"Sample '{self.sample.name}' is not a loop.")
        loop_beats = self._calculate_loop_bars() * self.grid.beats_per_bar
        return np.arange(0, self.grid.beats, loop_beats, dtype=float)

    def _calculate_loop_bars(self) -> int:
        length = len(self.sample.signal)
        if length < self.grid.bar:
            raise ValueError(
                f"Loop '{self.sample.name}' must be at least one bar long."
            )
        bars = 1
        while length >= 2 * bars * self.grid.bar:
            bars *= 2
        return bars


def draw_loop_track(
    grid: Grid, bank: SampleBank, layer: LayerConfig, rng: np.random.Generator
) -> Track:
    """A loop stretched to the grid tempo and repeated from the first bar."""
    track = Track(grid)
    sample = bank.draw(layer.kind, rng, is_loop=True).stretch(track)
    logger.debug(f"{layer.kind} loop {sample.name}")
    for beat in LoopPatternGenerator(grid, sample, rng=rng).generate():
        track.insert(sample.signal, beat, blend=False)
    return track
