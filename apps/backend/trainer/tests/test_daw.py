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
    Section,
    Sidechain,
    Trim,
    cut_signal,
    filter_signal,
    level_signal,
    loudest_rms,
    normalize_signal,
    section_signal,
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
        (
            Filter(500, "highpass", order=4, q=4.0),
            lambda s, sr: filter_signal(s, 500, "highpass", 4, q=4.0, sr=sr),
        ),
        (Normalize(), lambda s, sr: normalize_signal(s)),
        (
            Sidechain(np.array([0, 300]), ratio=0.7),
            lambda s, sr: sidechain_signal(
                s, np.array([0, 300]), ratio=0.7, sr=sr
            ),
        ),
        (
            Section(Gain(2.0), start=200, end=600),
            lambda s, sr: section_signal(s, Gain(2.0), 200, 600, sr=sr),
        ),
    ],
    ids=[
        "cut",
        "trim",
        "filter",
        "resonant_filter",
        "normalize",
        "sidechain",
        "section",
    ],
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


def test_loudest_rms_finds_the_loudest_window() -> None:
    signal = np.full(CONFIG.sr, 0.1, dtype=np.float32)
    signal[CONFIG.sr // 2 : CONFIG.sr // 2 + 4410] = 0.8

    assert loudest_rms(signal) == pytest.approx(0.8)


def test_loudest_rms_uses_the_whole_signal_when_shorter_than_window() -> None:
    signal = np.full(100, -0.5, dtype=np.float32)

    assert loudest_rms(signal) == pytest.approx(0.5)
    assert loudest_rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_level_signal_scales_to_the_level() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1, 1, CONFIG.sr).astype(np.float32)

    leveled = level_signal(signal, 0.25)

    assert leveled.dtype == np.float32
    assert loudest_rms(leveled) == pytest.approx(0.25, rel=1e-5)


def test_level_signal_leaves_silence_unchanged() -> None:
    silence = np.zeros(100, dtype=np.float32)

    np.testing.assert_array_equal(level_signal(silence, 0.5), silence)


def test_section_applies_the_effect_only_inside_the_section() -> None:
    sr = 1000
    signal = np.ones(1000, dtype=np.float32)

    result = section_signal(signal, Gain(3.0), 200, 600, fade=0.01, sr=sr)

    np.testing.assert_array_equal(result[:200], 1.0)
    np.testing.assert_array_equal(result[600:], 1.0)
    np.testing.assert_allclose(result[210:590], 3.0)
    assert np.all(np.diff(result[200:210]) > 0)
    assert np.all(np.diff(result[590:600]) < 0)


def test_section_keeps_the_effect_state_continuous() -> None:
    sr = 8000
    signal = (
        np.random.default_rng(0).uniform(-0.5, 0.5, 4000).astype(np.float32)
    )
    effect = Filter(500, "lowpass")

    result = section_signal(signal, effect, 1000, 3000, sr=sr)

    np.testing.assert_allclose(
        result[1040:2960], effect.process(signal, sr)[1040:2960], atol=1e-6
    )


def test_section_end_is_clipped_to_the_signal() -> None:
    signal = np.ones(100, dtype=np.float32)

    result = section_signal(signal, Gain(2.0), 50, 200, fade=0, sr=1000)

    np.testing.assert_array_equal(result[50:], 2.0)


@pytest.mark.parametrize(("start", "end"), [(-1, 50), (60, 50), (150, 200)])
def test_section_rejects_invalid_bounds(start: int, end: int) -> None:
    signal = np.ones(100, dtype=np.float32)

    with pytest.raises(ValueError):
        section_signal(signal, Gain(2.0), start, end)


def test_section_rejects_effects_that_change_the_length() -> None:
    signal = np.ones(1000, dtype=np.float32)

    with pytest.raises(ValueError, match="length"):
        section_signal(signal, Cut(500), 100, 200)


def test_resonant_filter_without_resonance_is_butterworth() -> None:
    sr = 8000
    signal = (
        np.random.default_rng(0).uniform(-0.5, 0.5, 2000).astype(np.float32)
    )

    for btype in ("lowpass", "highpass"):
        np.testing.assert_allclose(
            filter_signal(signal, 500, btype, q=2**-0.5, sr=sr),
            filter_signal(signal, 500, btype, sr=sr),
            atol=1e-6,
        )


def _sine_gain(
    frequency: float, cutoff: float, btype: str, order: int, q: float
) -> float:
    """The steady-state gain of the resonant filter for a sine."""
    sr = 8000
    time = np.arange(4 * sr) / sr
    sine = np.sin(2 * np.pi * frequency * time).astype(np.float32)
    filtered = filter_signal(sine, cutoff, btype, order, q=q, sr=sr)
    return float(np.abs(filtered[-sr:]).max())


@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("q", [0.7, 4.0, 10.0])
@pytest.mark.parametrize("order", [2, 4, 6])
def test_resonant_filter_gain_at_cutoff(
    btype: str, q: float, order: int
) -> None:
    expected = q if order == 2 else q * 2**-0.5

    assert _sine_gain(500, 500, btype, order, q) == pytest.approx(
        expected, rel=0.01
    )


@pytest.mark.parametrize(
    ("btype", "frequency"), [("lowpass", 1000), ("highpass", 250)]
)
def test_resonant_filter_slope_grows_with_order(
    btype: str, frequency: float
) -> None:
    q = 2**-0.5
    db_order_2 = 20 * np.log10(_sine_gain(frequency, 500, btype, 2, q))
    db_order_4 = 20 * np.log10(_sine_gain(frequency, 500, btype, 4, q))
    db_order_6 = 20 * np.log10(_sine_gain(frequency, 500, btype, 6, q))

    assert db_order_2 < -6
    assert db_order_4 == pytest.approx(2 * db_order_2, rel=0.01)
    assert db_order_6 == pytest.approx(3 * db_order_2, rel=0.02)


@pytest.mark.parametrize(
    ("cutoff", "btype", "order", "q"),
    [
        ((100, 500), "bandpass", 2, 2.0),
        (500, "bandpass", 2, 2.0),
        (5000, "lowpass", 2, 2.0),
        (500, "lowpass", 2, 0.0),
        (500, "lowpass", 3, 2.0),
        (500, "lowpass", 0, 2.0),
    ],
)
def test_resonant_filter_rejects_invalid_settings(
    cutoff: float | tuple[float, float], btype: str, order: int, q: float
) -> None:
    with pytest.raises(ValueError):
        filter_signal(
            np.ones(10, dtype=np.float32), cutoff, btype, order, q=q, sr=8000
        )
