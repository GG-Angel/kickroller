from dataclasses import dataclass

import numpy as np
import pytest

from src.core.config import CONFIG
from src.models.audio import Signal
from src.services.synthesis.effects import (
    Cut,
    Effect,
    Filter,
    Normalize,
    Sidechain,
    Trim,
    cut_signal,
    filter_signal,
    normalize_signal,
    sidechain_signal,
    trim_signal,
)
from src.services.synthesis.mix import Mix
from src.services.synthesis.track import Grid, Track


def make_track(signal: np.ndarray) -> Track:
    track = Track(Grid(bpm=60, bars=1, sr=CONFIG.sr))
    track.signal[: len(signal)] = signal
    return track


@dataclass(frozen=True)
class Gain(Effect):
    amount: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return signal * np.float32(self.amount)


@dataclass(frozen=True)
class Offset(Effect):
    amount: float

    def process(self, signal: Signal, sr: int) -> Signal:
        return signal + np.float32(self.amount)


def test_track_process_without_effects_returns_dry_copy() -> None:
    track = make_track(np.ones(16, dtype=np.float32))

    wet = track.process()

    np.testing.assert_array_equal(wet, track.signal)
    assert wet is not track.signal


def test_track_process_applies_effects_in_order() -> None:
    track = make_track(np.ones(16, dtype=np.float32))

    track.add_effects(Gain(2.0), Offset(1.0))
    track.add_effects(Gain(3.0))

    wet = track.process()

    assert wet[0] == pytest.approx((1.0 * 2.0 + 1.0) * 3.0)
    assert wet[-1] == pytest.approx((0.0 * 2.0 + 1.0) * 3.0)


def test_track_process_leaves_dry_signal_untouched() -> None:
    track = make_track(np.ones(16, dtype=np.float32))
    dry = track.signal.copy()
    track.add_effects(Sidechain(np.array([0]), ratio=1.0))

    first = track.process()
    second = track.process()

    np.testing.assert_array_equal(track.signal, dry)
    np.testing.assert_array_equal(first, second)


def test_mix_sums_wet_track_signals() -> None:
    dry_track = make_track(np.ones(16, dtype=np.float32))
    wet_track = make_track(np.ones(16, dtype=np.float32))
    wet_track.add_effects(Gain(0.5), Cut(8, fade=0.0))

    mix = Mix()
    mix.add_tracks(dry_track, wet_track)
    mixed = mix.mix()

    assert len(mixed) == dry_track.samples
    np.testing.assert_allclose(mixed[:8], 1.5)
    np.testing.assert_allclose(mixed[8:16], 1.0)


@pytest.mark.parametrize(
    ("effect", "function"),
    [
        (Cut(100), lambda s, sr: cut_signal(s, 100, sr=sr)),
        (Trim(), lambda s, sr: trim_signal(s)),
        (
            Filter(500, "lowpass"),
            lambda s, sr: filter_signal(s, 500, "lowpass", sr=sr),
        ),
        (Normalize(), lambda s, sr: normalize_signal(s)),
        (
            Sidechain(np.array([0, 300]), ratio=0.7),
            lambda s, sr: sidechain_signal(
                s, np.array([0, 300]), ratio=0.7, sr=sr
            ),
        ),
    ],
    ids=["cut", "trim", "filter", "normalize", "sidechain"],
)
def test_effects_delegate_to_functions(effect: Effect, function) -> None:
    sr = 8000
    signal = (
        np.random.default_rng(0).uniform(-0.5, 0.5, 1000).astype(np.float32)
    )

    np.testing.assert_array_equal(
        effect.process(signal, sr), function(signal, sr)
    )


def test_sidechain_ducks_at_each_trigger_for_requested_duration() -> None:
    signal = np.ones(CONFIG.sr, dtype=np.float32)
    triggers = np.array([0, CONFIG.sr // 2])

    wet = sidechain_signal(signal, triggers, ratio=0.5)

    assert wet.dtype == np.float32
    assert wet[0] == pytest.approx(1.0)
    assert wet[CONFIG.sr // 200] == pytest.approx(0.5, abs=1e-3)
    assert wet[CONFIG.sr // 10 - 1] == pytest.approx(1.0, abs=1e-3)
    assert wet[CONFIG.sr // 10] == pytest.approx(1.0)
    assert wet[CONFIG.sr // 2] == pytest.approx(1.0)
    assert wet[CONFIG.sr // 2 + CONFIG.sr // 10] == pytest.approx(1.0)


def test_sidechain_fades_smoothly_at_trigger_boundaries() -> None:
    signal = np.ones(CONFIG.sr, dtype=np.float32)
    start = 1000
    end = start + CONFIG.sr // 10

    wet = sidechain_signal(signal, np.array([start]), ratio=1.0)

    assert wet[start] == pytest.approx(1.0)
    assert wet[start + 1] < 1.0
    assert wet[end - 1] > 0.99
    assert wet[end] == pytest.approx(1.0)


def test_sidechain_does_not_stack_overlapping_triggers() -> None:
    signal = np.ones(1000, dtype=np.float32)

    wet = sidechain_signal(signal, np.array([10, 15]), ratio=0.5, duration=0.01)

    assert wet[235] == pytest.approx(0.5)


def apply_sidechain(
    triggers: np.ndarray, ratio: float = 0.5, duration: float = 0.1
) -> None:
    sidechain_signal(np.ones(16, dtype=np.float32), triggers, ratio, duration)


def process_sidechain(
    triggers: np.ndarray, ratio: float = 0.5, duration: float = 0.1
) -> None:
    Sidechain(triggers, ratio, duration).process(
        np.ones(16, dtype=np.float32), CONFIG.sr
    )


SIDECHAIN_ENTRY_POINTS = pytest.mark.parametrize(
    "make", [apply_sidechain, process_sidechain], ids=["function", "effect"]
)


@SIDECHAIN_ENTRY_POINTS
@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_sidechain_rejects_invalid_ratio(make, ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
        make(np.array([0]), ratio=ratio)


@SIDECHAIN_ENTRY_POINTS
@pytest.mark.parametrize("duration", [-0.1, float("nan"), float("inf")])
def test_sidechain_rejects_invalid_duration(make, duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be"):
        make(np.array([0]), duration=duration)


@SIDECHAIN_ENTRY_POINTS
def test_sidechain_rejects_negative_trigger(make) -> None:
    with pytest.raises(ValueError, match="non-negative sample indices"):
        make(np.array([-1]))
