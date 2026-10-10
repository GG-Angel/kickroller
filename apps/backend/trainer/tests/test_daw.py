import numpy as np
import pytest

from src.core.config import CONFIG
from src.services.synthesis.track import Grid, Track


def make_track(signal: np.ndarray) -> Track:
    track = Track(Grid(bpm=60, bars=1, sr=CONFIG.sr))
    track.signal[: len(signal)] = signal
    return track


def test_sidechain_ducks_at_each_trigger_for_requested_duration() -> None:
    track = make_track(np.ones(CONFIG.sr, dtype=np.float32))
    triggers = np.array([0, CONFIG.sr // 2])

    track.sidechain(triggers, ratio=0.5)

    assert track.signal.dtype == np.float32
    assert track.signal[0] == pytest.approx(1.0)
    assert track.signal[CONFIG.sr // 200] == pytest.approx(0.5, abs=1e-3)
    assert track.signal[CONFIG.sr // 10 - 1] == pytest.approx(1.0, abs=1e-3)
    assert track.signal[CONFIG.sr // 10] == pytest.approx(1.0)
    assert track.signal[CONFIG.sr // 2] == pytest.approx(1.0)
    assert track.signal[CONFIG.sr // 2 + CONFIG.sr // 10] == pytest.approx(1.0)


def test_sidechain_fades_smoothly_at_trigger_boundaries() -> None:
    track = make_track(np.ones(CONFIG.sr, dtype=np.float32))
    start = 1000
    end = start + CONFIG.sr // 10

    track.sidechain(np.array([start]), ratio=1.0)

    assert track.signal[start] == pytest.approx(1.0)
    assert track.signal[start + 1] < 1.0
    assert track.signal[end - 1] > 0.99
    assert track.signal[end] == pytest.approx(1.0)


def test_sidechain_does_not_stack_overlapping_triggers() -> None:
    track = make_track(np.ones(1000, dtype=np.float32))

    track.sidechain(np.array([10, 15]), ratio=0.5, duration=0.01)

    assert track.signal[235] == pytest.approx(0.5)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_sidechain_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
        make_track(np.ones(16, dtype=np.float32)).sidechain(np.array([0]), ratio)


@pytest.mark.parametrize("duration", [-0.1, float("nan"), float("inf")])
def test_sidechain_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be"):
        make_track(np.ones(16, dtype=np.float32)).sidechain(
            np.array([0]), duration=duration
        )


def test_sidechain_rejects_negative_trigger() -> None:
    with pytest.raises(ValueError, match="non-negative sample indices"):
        make_track(np.ones(16, dtype=np.float32)).sidechain(np.array([-1]))
