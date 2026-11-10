from dataclasses import dataclass

import numpy as np
import pytest

from src.core.config import CONFIG
from src.models.audio import Signal
from src.services.synthesis import effects
from src.services.synthesis.effects import (
    Clip,
    Cut,
    Effect,
    Filter,
    Level,
    Limit,
    Mp3,
    Normalize,
    Section,
    Sidechain,
    Tilt,
    Trim,
    loudest_rms,
)
from src.services.synthesis.effects import (
    Gain as AudioGain,
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

    mix = Mix(CONFIG.sr)
    mix.add_tracks(dry_track, wet_track)
    mixed = mix.mix()

    assert len(mixed) == dry_track.samples
    np.testing.assert_allclose(mixed[:8], 1.5)
    np.testing.assert_allclose(mixed[8:16], 1.0)


def test_trim_removes_silence_around_the_signal() -> None:
    signal = np.concatenate(
        [
            np.zeros(4000, dtype=np.float32),
            np.ones(8000, dtype=np.float32),
            np.zeros(4000, dtype=np.float32),
        ]
    )

    trimmed = Trim().process(signal, 8000)

    assert 0 < len(trimmed) < len(signal)
    assert trimmed.dtype == np.float32


def test_sidechain_ducks_at_each_trigger_for_requested_duration() -> None:
    signal = np.ones(CONFIG.sr, dtype=np.float32)
    triggers = np.array([0, CONFIG.sr // 2])

    wet = Sidechain(triggers, ratio=0.5).process(signal, CONFIG.sr)

    assert wet.dtype == np.float32
    assert wet[0] == pytest.approx(1.0)
    assert wet[CONFIG.sr // 200] == pytest.approx(0.5, abs=1e-3)
    assert 0.5 < wet[CONFIG.sr // 20] < 1.0
    assert wet[CONFIG.sr // 10 - 1] == pytest.approx(1.0, abs=1e-3)
    assert wet[CONFIG.sr // 10] == pytest.approx(1.0)
    assert wet[CONFIG.sr // 2] == pytest.approx(1.0)
    assert wet[CONFIG.sr // 2 + CONFIG.sr // 10] == pytest.approx(1.0)


def test_sidechain_fades_smoothly_at_trigger_boundaries() -> None:
    signal = np.ones(CONFIG.sr, dtype=np.float32)
    start = 1000
    end = start + CONFIG.sr // 10

    wet = Sidechain(np.array([start]), ratio=1.0).process(signal, CONFIG.sr)

    assert wet[start] == pytest.approx(1.0)
    assert wet[start + 1] < 1.0
    assert wet[end - 1] > 0.99
    assert wet[end] == pytest.approx(1.0)


def test_sidechain_does_not_stack_overlapping_triggers() -> None:
    signal = np.ones(1000, dtype=np.float32)

    wet = Sidechain(np.array([10, 15]), ratio=0.5, duration=0.01).process(
        signal, CONFIG.sr
    )

    assert wet[235] == pytest.approx(0.5)


def process_sidechain(
    triggers: np.ndarray, ratio: float = 0.5, duration: float = 0.1
) -> None:
    Sidechain(triggers, ratio, duration).process(
        np.ones(16, dtype=np.float32), CONFIG.sr
    )


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_sidechain_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
        process_sidechain(np.array([0]), ratio=ratio)


@pytest.mark.parametrize("duration", [-0.1, float("nan"), float("inf")])
def test_sidechain_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be"):
        process_sidechain(np.array([0]), duration=duration)


def test_sidechain_rejects_negative_trigger() -> None:
    with pytest.raises(ValueError, match="non-negative sample indices"):
        process_sidechain(np.array([-1]))


def test_loudest_rms_finds_the_loudest_window() -> None:
    signal = np.full(CONFIG.sr, 0.1, dtype=np.float32)
    signal[CONFIG.sr // 2 : CONFIG.sr // 2 + 4410] = 0.8

    assert loudest_rms(signal, CONFIG.sr) == pytest.approx(0.8)


def test_loudest_rms_uses_the_whole_signal_when_shorter_than_window() -> None:
    signal = np.full(100, -0.5, dtype=np.float32)

    assert loudest_rms(signal, CONFIG.sr) == pytest.approx(0.5)
    assert loudest_rms(np.zeros(0, dtype=np.float32), CONFIG.sr) == 0.0


def test_level_scales_to_the_level() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(-1, 1, CONFIG.sr).astype(np.float32)

    leveled = Level(0.25).process(signal, CONFIG.sr)

    assert leveled.dtype == np.float32
    assert loudest_rms(leveled, CONFIG.sr) == pytest.approx(0.25, rel=1e-5)


def test_level_leaves_silence_unchanged() -> None:
    silence = np.zeros(100, dtype=np.float32)

    np.testing.assert_array_equal(
        Level(0.5).process(silence, CONFIG.sr), silence
    )


def test_section_applies_the_effect_only_inside_the_section() -> None:
    sr = 1000
    signal = np.ones(1000, dtype=np.float32)

    result = Section(Gain(3.0), 200, 600, fade=0.01).process(signal, sr)

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

    result = Section(effect, 1000, 3000).process(signal, sr)

    np.testing.assert_allclose(
        result[1040:2960], effect.process(signal, sr)[1040:2960], atol=1e-6
    )


def test_section_end_is_clipped_to_the_signal() -> None:
    signal = np.ones(100, dtype=np.float32)

    result = Section(Gain(2.0), 50, 200, fade=0).process(signal, 1000)

    np.testing.assert_array_equal(result[50:], 2.0)


@pytest.mark.parametrize(("start", "end"), [(-1, 50), (60, 50), (150, 200)])
def test_section_rejects_invalid_bounds(start: int, end: int) -> None:
    signal = np.ones(100, dtype=np.float32)

    with pytest.raises(ValueError):
        Section(Gain(2.0), start, end).process(signal, CONFIG.sr)


def test_section_rejects_effects_that_change_the_length() -> None:
    signal = np.ones(1000, dtype=np.float32)

    with pytest.raises(ValueError, match="length"):
        Section(Cut(500), 100, 200).process(signal, CONFIG.sr)


def test_resonant_filter_without_resonance_is_butterworth() -> None:
    sr = 8000
    signal = (
        np.random.default_rng(0).uniform(-0.5, 0.5, 2000).astype(np.float32)
    )

    for btype in ("lowpass", "highpass"):
        np.testing.assert_allclose(
            Filter(500, btype, q=2**-0.5).process(signal, sr),
            Filter(500, btype).process(signal, sr),
            atol=1e-6,
        )


def _sine_gain(
    frequency: float, cutoff: float, btype: str, order: int, q: float
) -> float:
    """The steady-state gain of the resonant filter for a sine."""
    sr = 8000
    time = np.arange(4 * sr) / sr
    sine = np.sin(2 * np.pi * frequency * time).astype(np.float32)
    filtered = Filter(cutoff, btype, order, q).process(sine, sr)
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
        Filter(cutoff, btype, order, q).process(
            np.ones(10, dtype=np.float32), 8000
        )


def _click_train(seconds: float = 2.0) -> Signal:
    rng = np.random.default_rng(0)
    signal = 0.05 * rng.standard_normal(round(seconds * CONFIG.sr))
    signal[CONFIG.sr // 4 :: CONFIG.sr // 3] = 0.9
    return signal.astype(np.float32)


def _lag(wet: Signal, dry: Signal, max_lag: int = 2000) -> int:
    lags = np.arange(-max_lag, max_lag + 1)
    scores = [np.dot(np.roll(dry, lag), wet) for lag in lags]
    return int(lags[np.argmax(scores)])


def test_normalize_sets_the_peak() -> None:
    signal = _click_train()

    normalized = Normalize(-6.0).process(signal, CONFIG.sr)

    assert np.abs(normalized).max() == pytest.approx(10 ** (-6 / 20))
    assert normalized.dtype == np.float32


def test_gain_and_tilt_of_zero_db_keep_the_signal() -> None:
    signal = _click_train()

    np.testing.assert_allclose(
        AudioGain(0.0).process(signal, CONFIG.sr), signal
    )
    np.testing.assert_allclose(
        Tilt(0.0).process(signal, CONFIG.sr), signal, atol=1e-6
    )


def test_tilt_moves_the_balance_of_lows_and_highs() -> None:
    t = np.arange(CONFIG.sr) / CONFIG.sr
    low = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

    def gain_db(signal: Signal) -> float:
        tilted = Tilt(6.0).process(signal, CONFIG.sr)[CONFIG.sr // 2 :]
        return 20 * np.log10(np.abs(tilted).max() / np.abs(signal).max())

    assert gain_db(low) == pytest.approx(-3.0, abs=0.1)
    assert gain_db(high) == pytest.approx(3.0, abs=0.1)


def test_clip_stays_at_or_below_one() -> None:
    clipped = Clip(12.0).process(_click_train() * 4, CONFIG.sr)

    assert np.abs(clipped).max() <= 1.0


def test_limit_keeps_the_ceiling_and_the_timing() -> None:
    signal = _click_train()

    limited = Limit(9.0, ceiling_db=-1.0).process(signal, CONFIG.sr)

    assert np.abs(limited).max() <= 10 ** (-1 / 20) + 1e-6
    assert _lag(limited, signal) == 0
    quiet = Limit(0.0).process(signal * 0.1, CONFIG.sr)
    np.testing.assert_allclose(quiet, signal * 0.1, atol=1e-7)


def test_mp3_keeps_the_length_and_the_timing() -> None:
    signal = _click_train()

    encoded = Mp3(0.75).process(signal, CONFIG.sr)

    assert len(encoded) == len(signal)
    assert encoded.dtype == np.float32
    assert _lag(encoded, signal) == 0
    assert not np.array_equal(encoded, signal)


@pytest.mark.parametrize(
    "effect",
    [effects.Gain(-6.0), Tilt(6.0), Clip(6.0), Limit(6.0), Mp3(0.5)],
)
def test_master_effects_keep_the_length_and_dtype(effect: Effect) -> None:
    signal = _click_train(0.5)

    wet = effect.process(signal, CONFIG.sr)

    assert len(wet) == len(signal)
    assert wet.dtype == np.float32
    assert np.isfinite(wet).all()


def test_mix_applies_master_effects_after_the_sum() -> None:
    mix = Mix(CONFIG.sr)
    mix.add_tracks(
        make_track(np.full(4, 0.5, dtype=np.float32)),
        make_track(np.full(4, 0.5, dtype=np.float32)),
    )
    mix.add_effects(Offset(1.0), Gain(2.0))

    mixed = mix.mix()

    assert mixed[0] == pytest.approx((0.5 + 0.5 + 1.0) * 2.0)
