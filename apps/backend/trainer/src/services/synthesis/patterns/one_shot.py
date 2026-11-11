from dataclasses import dataclass, field

import librosa
import numpy as np
from loguru import logger

from src.core.config import LayerConfig
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.daw.effects import Level, Trim
from src.services.synthesis.daw.grid import Grid
from src.services.synthesis.daw.track import Track


@dataclass
class OneShotPatternGenerator:
    """Draw one-shots at regular beat positions."""

    grid: Grid
    """Grid the pattern is generated for."""
    period: float
    """Beats between the allowed positions."""
    offset: float = 0.0
    """Beat of the first allowed position."""
    probability: float = 1.0
    """Chance that each allowed position plays, between 0 and 1."""
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    """Random number generator used to draw the pattern."""

    def generate(self) -> np.ndarray:
        """Draw one-shot positions. Raise ValueError for invalid settings."""
        if self.period <= 0:
            raise ValueError("Period must be positive.")
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        beats = np.arange(self.offset, self.grid.beats, self.period)
        return beats[self.rng.random(len(beats)) < self.probability]


def draw_one_shot_track(
    grid: Grid,
    bank: SampleBank,
    layer: LayerConfig,
    kick_level: float,
    rng: np.random.Generator,
) -> Track:
    """Draw and level one-shots at a layer placement. Each cuts the prior tail."""
    track = Track(grid)
    sample = bank.draw(layer.kind, rng, is_loop=False)
    placement = layer.placements[rng.integers(len(layer.placements))]
    period = (
        1 / grid.steps_per_beat
        if placement.period is None
        else placement.period
    )
    probability = rng.uniform(*placement.probability_range)
    beats = OneShotPatternGenerator(
        grid, period, placement.offset, probability, rng=rng
    ).generate()
    db = rng.uniform(*layer.level_db_range)
    level = kick_level * float(librosa.db_to_amplitude(db))
    trimmed = Trim().process(sample.signal, grid.sr)
    signal = Level(level).process(trimmed, grid.sr)
    logger.debug(
        f"{layer.kind} one-shot {sample.name} at {db:.1f} dB: {len(beats)} "
        f"hits, every {period:g} beats from {placement.offset:g}"
    )
    for beat in beats:
        track.insert(signal, beat, blend=False)
    return track
