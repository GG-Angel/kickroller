from pathlib import Path

import numpy as np
import pytest

from src.services.synthesis.bank import Sample, SampleKind
from src.services.synthesis.synth import MelodyPatternGenerator
from src.services.synthesis.track import Grid

# 1 beat = 60 samples, 1 bar = 240 samples, 1 step = 15 samples
GRID = Grid(bpm=60, bars=12, sr=60)


def make_loop(samples: int) -> Sample:
    return Sample(
        name="melody",
        path=Path("melody.wav"),
        signal=np.zeros(samples, dtype=np.float32),
        kind=SampleKind.MELODY,
        bpm=GRID.bpm,
        is_loop=True,
    )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        (4 * 240, [0, 16, 32]),
        (4 * 240 + 120, [0, 16, 32]),
        (4 * 240 - 5, [0, 16, 32]),
        (8 * 240, [0, 32]),
        (8 * 240 + 600, [0, 32]),
        (2 * 240 + 30, [0, 8, 16, 24, 32, 40]),
    ],
    ids=[
        "4-bars-exact",
        "4-bars-with-tail",
        "4-bars-slightly-short",
        "8-bars-exact",
        "8-bars-with-tail",
        "2-bars-with-tail",
    ],
)
def test_melody_pattern_repeats_loop_until_drop_ends(
    samples: int, expected: list[int]
) -> None:
    pattern = MelodyPatternGenerator(GRID, make_loop(samples)).generate()

    np.testing.assert_array_equal(pattern, expected)


def test_melody_pattern_rejects_loop_shorter_than_a_bar() -> None:
    with pytest.raises(ValueError, match="at least one bar long"):
        MelodyPatternGenerator(GRID, make_loop(200)).generate()
