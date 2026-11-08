import numpy as np
import pytest

from src.models.audio import LabeledDrop
from src.services.synthesis.crop import CropWindow, crop_drop, draw_crop

SR = 100


def _drop() -> LabeledDrop:
    return LabeledDrop(
        signal=np.arange(1000, dtype=np.float32),
        onsets=np.array([0.0, 1.0, 2.5, 4.0, 9.99]),
    )


def test_crop_drop_cuts_signal_and_shifts_onsets() -> None:
    cropped = crop_drop(_drop(), CropWindow(start=1.0, end=4.0), sr=SR)

    np.testing.assert_array_equal(
        cropped.signal, np.arange(100, 400, dtype=np.float32)
    )
    np.testing.assert_allclose(cropped.onsets, [0.0, 1.5])


def test_crop_drop_rounds_times_to_samples() -> None:
    cropped = crop_drop(_drop(), CropWindow(start=0.994, end=2.0), sr=SR)

    assert cropped.signal[0] == 99
    np.testing.assert_allclose(cropped.onsets, [0.01])


@pytest.mark.parametrize(
    ("start", "end"), [(-0.5, 2.0), (2.0, 2.0), (3.0, 2.0), (5.0, 10.5)]
)
def test_crop_drop_rejects_windows_outside_the_drop(
    start: float, end: float
) -> None:
    with pytest.raises(ValueError):
        crop_drop(_drop(), CropWindow(start=start, end=end), sr=SR)


def test_draw_crop_stays_inside_the_duration() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        window = draw_crop(rng, duration=13.0, seconds=12.0)
        assert 0 <= window.start <= 1.0
        assert window.end == pytest.approx(window.start + 12.0)


def test_draw_crop_rejects_short_drops() -> None:
    with pytest.raises(ValueError):
        draw_crop(np.random.default_rng(0), duration=11.0, seconds=12.0)
