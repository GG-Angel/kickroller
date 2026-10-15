import numpy as np

from src.core.config import CONFIG
from src.services.synthesis.synth import draw_grid


def test_draw_grid_is_one_bar_longer_than_the_crop() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        grid = draw_grid(CONFIG.synth, rng)
        seconds = grid.bars * grid.bar / grid.sr
        assert seconds >= CONFIG.synth.drop_seconds + grid.bar / grid.sr
