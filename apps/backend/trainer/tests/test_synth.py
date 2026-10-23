from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.config import (
    CONFIG,
    KickFxConfig,
    KickRollConfig,
    LayerConfig,
    MasterConfig,
    Placement,
    SectionConfig,
)
from src.models.audio import SampleKind
from src.services.synthesis.bank import Sample, SampleBank
from src.services.synthesis.effects import (
    Clip,
    Filter,
    Gain,
    Limit,
    Mp3,
    Normalize,
    Section,
    Sidechain,
    Tilt,
    loudest_rms,
)
from src.services.synthesis.grid import Grid
from src.services.synthesis.master import draw_master_effects
from src.services.synthesis.patterns.kick import draw_kick_fx, draw_kick_track
from src.services.synthesis.patterns.loop import draw_loop_track
from src.services.synthesis.patterns.one_shot import (
    OneShotPatternGenerator,
    draw_one_shot_track,
)
from src.services.synthesis.synth import (
    _add_layer_effects,
    draw_grid,
    generate_drop,
)
from src.services.synthesis.track import Track


def test_draw_grid_is_one_bar_longer_than_the_crop() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        grid = draw_grid(CONFIG.synth, rng)
        seconds = grid.bars * grid.bar / grid.sr
        assert seconds >= CONFIG.synth.drop_seconds + grid.bar / grid.sr


def test_one_shot_pattern_uses_period_and_offset() -> None:
    grid = Grid(bpm=160, bars=2)

    beats = OneShotPatternGenerator(grid, period=2, offset=1).generate()

    np.testing.assert_array_equal(beats, [1, 3, 5, 7])


def test_one_shot_pattern_drops_positions_by_probability() -> None:
    grid = Grid(bpm=160, bars=64)
    rng = np.random.default_rng(0)

    beats = OneShotPatternGenerator(
        grid, period=1, probability=0.25, rng=rng
    ).generate()

    assert np.all(beats == np.round(beats))
    assert 0.2 < len(beats) / grid.beats < 0.3


@pytest.mark.parametrize(
    ("period", "probability"), [(0, 0.5), (-1, 0.5), (1, 1.5)]
)
def test_one_shot_pattern_rejects_invalid_settings(
    period: float, probability: float
) -> None:
    generator = OneShotPatternGenerator(
        Grid(bpm=160, bars=1), period=period, probability=probability
    )

    with pytest.raises(ValueError):
        generator.generate()


def test_one_shot_layer_needs_placements() -> None:
    with pytest.raises(ValidationError, match="no placements"):
        LayerConfig(
            kind=SampleKind.FX,
            is_loop=False,
            track_count_weights={1: 1.0},
            level_db_range=(0, 0),
        )


def _sample(kind: SampleKind, is_loop: bool, seconds: float) -> Sample:
    rng = np.random.default_rng(len(kind))
    signal = rng.uniform(-0.5, 0.5, round(seconds * CONFIG.sr))
    return Sample(
        name=f"{kind}_{'loop' if is_loop else 'one_shot'}",
        path=Path("unused.wav"),
        signal=signal.astype(np.float32),
        kind=kind,
        bpm=160,
        is_loop=is_loop,
    )


def _bank() -> SampleBank:
    one_shots = [_sample(kind, False, 0.3) for kind in SampleKind]
    loops = [
        _sample(kind, True, 6.0)  # 4 bars at 160 BPM
        for kind in (SampleKind.MELODY, SampleKind.VOCAL, SampleKind.DRUM)
    ]
    return SampleBank(samples=one_shots + loops)


def test_sample_bank_draws_only_loops_or_one_shots() -> None:
    bank = _bank()
    rng = np.random.default_rng(0)

    assert bank.draw(SampleKind.DRUM, rng, is_loop=True).is_loop
    assert not bank.draw(SampleKind.DRUM, rng, is_loop=False).is_loop
    with pytest.raises(ValueError, match="no fx loop samples"):
        bank.draw(SampleKind.FX, rng, is_loop=True)


def test_layers_do_not_change_kick_onsets() -> None:
    config = CONFIG.synth.model_copy(update={"bpms": [160]})
    every_layer = [
        layer.model_copy(update={"track_count_weights": {1: 1.0}})
        for layer in config.layers
    ]
    bank = _bank()

    kicks_only = generate_drop(
        bank,
        np.random.default_rng(3),
        config.model_copy(update={"layers": []}),
    )
    full = generate_drop(
        bank,
        np.random.default_rng(3),
        config.model_copy(update={"layers": every_layer}),
    )

    np.testing.assert_array_equal(full.onsets, kicks_only.onsets)
    assert len(full.signal) == len(kicks_only.signal)
    assert not np.array_equal(full.signal, kicks_only.signal)


@pytest.mark.parametrize("db", [0.0, -6.0])
def test_layer_tracks_are_leveled_relative_to_the_kick(db: float) -> None:
    grid = Grid(bpm=160, bars=4)
    bank = _bank()
    rng = np.random.default_rng(0)
    one_shot_layer = LayerConfig(
        kind=SampleKind.DRUM,
        is_loop=False,
        track_count_weights={1: 1.0},
        level_db_range=(db, db),
        placements=[Placement(period=4)],
    )
    loop_layer = one_shot_layer.model_copy(
        update={"is_loop": True, "placements": []}
    )

    one_shot = draw_one_shot_track(grid, bank, one_shot_layer, 0.5, rng)
    loop = draw_loop_track(grid, bank, loop_layer, 0.5, rng)

    expected = 0.5 * 10 ** (db / 20)
    assert loudest_rms(one_shot.signal) == pytest.approx(expected, rel=1e-4)
    assert loudest_rms(loop.signal) == pytest.approx(expected, rel=1e-4)


def test_layer_high_pass_cutoff_is_drawn_from_its_range_or_off() -> None:
    layer = LayerConfig(
        kind=SampleKind.MELODY,
        is_loop=True,
        track_count_weights={1: 1.0},
        level_db_range=(0, 0),
        high_pass_hz_range=(60.0, 200.0),
    )
    sidechain = Sidechain(np.array([0]))
    rng = np.random.default_rng(0)

    cutoffs = []
    for _ in range(50):
        track = Track(Grid(bpm=160, bars=1))
        _add_layer_effects(track, layer, sidechain, rng)
        (high_pass,) = track.effects
        assert isinstance(high_pass, Filter)
        assert isinstance(high_pass.cutoff, float)
        cutoffs.append(high_pass.cutoff)
    assert 60.0 <= min(cutoffs) < max(cutoffs) <= 200.0

    track = Track(Grid(bpm=160, bars=1))
    unfiltered = layer.model_copy(update={"high_pass_hz_range": None})
    _add_layer_effects(track, unfiltered, sidechain, rng)
    assert track.effects == []


def test_kick_fx_sections_are_whole_beats_and_do_not_overlap() -> None:
    grid = Grid(bpm=160, bars=2)
    config = KickFxConfig(section_count_weights={4: 1.0})
    rng = np.random.default_rng(0)

    for _ in range(50):
        sections = draw_kick_fx(grid, config, rng)
        beat_starts = set(grid.to_samples(np.arange(grid.beats + 1)))
        covered = np.zeros(grid.to_sample(grid.beats), dtype=int)
        for section in sections:
            assert section.start in beat_starts and section.end in beat_starts
            beats = round((section.end - section.start) / grid.beat)
            assert beats in config.section_beats
            covered[section.start : section.end] += 1
        assert covered.max() <= 1


def test_kick_fx_are_high_passes_inside_their_ranges() -> None:
    grid = Grid(bpm=160, bars=8)
    config = KickFxConfig(section_count_weights={2: 1.0})
    rng = np.random.default_rng(0)

    orders = set()
    for _ in range(50):
        for section in draw_kick_fx(grid, config, rng):
            assert isinstance(section.effect, Filter)
            assert section.effect.btype == "highpass"
            low, high = config.high_pass_hz_range
            cutoff = section.effect.cutoff
            assert isinstance(cutoff, float) and low <= cutoff <= high
            q = section.effect.q
            assert q is not None
            assert (
                config.resonance_q_range[0] <= q <= config.resonance_q_range[1]
            )
            orders.add(section.effect.order)
    assert orders == set(config.filter_orders)


def test_kick_fx_can_be_off() -> None:
    config = KickFxConfig(section_count_weights={0: 1.0})

    assert (
        draw_kick_fx(Grid(bpm=160, bars=2), config, np.random.default_rng())
        == []
    )


_NO_ROLLS = KickRollConfig(section_count_weights={0: 1.0})
_ROLLS = KickRollConfig(section_count_weights={2: 1.0})
_NO_OFF_BEATS = (0.0, 0.0)


def test_kick_off_beat_chance_adds_off_beat_kicks_on_any_grid() -> None:
    grid = Grid(bpm=160, bars=64, steps_per_beat=3)
    no_sections = SectionConfig(section_count_weights={0: 1.0})
    no_fx = KickFxConfig(section_count_weights={0: 1.0})

    kick = draw_kick_track(
        grid,
        _bank(),
        no_sections,
        _NO_ROLLS,
        no_fx,
        (0.5, 0.5),
        np.random.default_rng(0),
    )

    off_beats = np.isclose(kick.beats % 1, 0.5)
    assert 0.4 < off_beats.sum() / grid.beats < 0.6
    assert np.all(np.diff(kick.beats) > 0)


def test_kick_gaps_have_no_kicks_and_no_kick_fx() -> None:
    grid = Grid(bpm=160, bars=4)
    gap_config = SectionConfig(section_count_weights={2: 1.0})
    fx_config = KickFxConfig(section_count_weights={4: 1.0})
    rng = np.random.default_rng(0)

    for _ in range(50):
        kick = draw_kick_track(
            grid, _bank(), gap_config, _ROLLS, fx_config, _NO_OFF_BEATS, rng
        )

        assert len(kick.gaps) == 2
        for gap in kick.gaps:
            assert gap.end - gap.start in gap_config.section_beats
            assert not np.any(
                (kick.beats >= gap.start) & (kick.beats < gap.end)
            )
            gap_start, gap_end = (
                grid.to_sample(gap.start),
                grid.to_sample(gap.end),
            )
            for section in kick.track.effects:
                assert isinstance(section, Section)
                assert section.end <= gap_start or section.start >= gap_end


def test_kick_gaps_keep_the_kicks_outside_the_gaps() -> None:
    grid = Grid(bpm=160, bars=4)
    fx_config = KickFxConfig(section_count_weights={0: 1.0})
    no_gaps = SectionConfig(section_count_weights={0: 1.0})
    gaps = SectionConfig(section_count_weights={2: 1.0})

    full = draw_kick_track(
        grid,
        _bank(),
        no_gaps,
        _NO_ROLLS,
        fx_config,
        _NO_OFF_BEATS,
        np.random.default_rng(1),
    )
    gapped = draw_kick_track(
        grid,
        _bank(),
        gaps,
        _NO_ROLLS,
        fx_config,
        _NO_OFF_BEATS,
        np.random.default_rng(1),
    )

    outside = np.ones(len(full.beats), dtype=bool)
    for gap in gapped.gaps:
        outside &= (full.beats < gap.start) | (full.beats >= gap.end)
    np.testing.assert_array_equal(gapped.beats, full.beats[outside])


def test_kick_rolls_play_every_roll_step_at_the_roll_level() -> None:
    grid = Grid(bpm=160, bars=4)
    no_gaps = SectionConfig(section_count_weights={0: 1.0})
    rolls = KickRollConfig(
        section_count_weights={1: 1.0},
        section_beats=[2],
        steps_per_beat=[3],
        level_db_range=(-6, -6),
    )
    no_fx = KickFxConfig(section_count_weights={0: 1.0})

    kick = draw_kick_track(
        grid,
        _bank(),
        no_gaps,
        rolls,
        no_fx,
        _NO_OFF_BEATS,
        np.random.default_rng(0),
    )

    (roll,) = kick.rolls
    inside = (kick.beats >= roll.start) & (kick.beats < roll.end)
    np.testing.assert_allclose(
        kick.beats[inside], roll.start + np.arange(6) / 3
    )
    assert np.all(np.diff(kick.beats) > 0)
    start, end = grid.to_sample(roll.start), grid.to_sample(roll.end)
    roll_peak = np.abs(kick.track.signal[start:end]).max()
    other_peak = np.abs(np.delete(kick.track.signal, np.s_[start:end])).max()
    assert roll_peak / other_peak == pytest.approx(10 ** (-6 / 20), rel=0.01)


def test_master_effects_without_config_only_set_the_peak() -> None:
    effects = draw_master_effects(None, -1.0, np.random.default_rng(0))

    assert effects == [Normalize(-1.0)]


def test_master_effects_are_drawn_in_order_inside_their_ranges() -> None:
    config = MasterConfig(
        tilt_probability=1,
        clip_probability=1,
        limit_probability=1,
        mp3_probability=1,
    )

    effects = draw_master_effects(config, -1.0, np.random.default_rng(0))

    assert [type(effect) for effect in effects] == [
        Normalize,
        Tilt,
        Clip,
        Limit,
        Mp3,
        Gain,
    ]
    _, tilt, clip, limit, mp3, gain = effects
    assert isinstance(tilt, Tilt) and -6 <= tilt.db <= 6
    assert isinstance(clip, Clip) and 0 <= clip.drive_db <= 12
    assert isinstance(limit, Limit) and limit.ceiling_db == -0.3
    assert isinstance(mp3, Mp3) and 0 <= mp3.compression_level <= 0.75
    assert isinstance(gain, Gain) and -6 <= gain.db <= 6


def test_master_effects_can_skip_the_optional_effects() -> None:
    config = MasterConfig(
        tilt_probability=0,
        clip_probability=0,
        limit_probability=0,
        mp3_probability=0,
    )

    effects = draw_master_effects(config, -1.0, np.random.default_rng(0))

    assert [type(effect) for effect in effects] == [Normalize, Gain]


def test_generate_drop_without_master_peaks_at_peak_db() -> None:
    config = CONFIG.synth.model_copy(update={"master": None})

    drop = generate_drop(_bank(), np.random.default_rng(0), config)

    peak = 10 ** (config.peak_db / 20)
    assert np.abs(drop.signal).max() == pytest.approx(peak, rel=1e-5)


def test_generate_drop_with_master_is_the_same_for_a_seed() -> None:
    first = generate_drop(_bank(), np.random.default_rng(4))
    second = generate_drop(_bank(), np.random.default_rng(4))

    np.testing.assert_array_equal(first.signal, second.signal)
    np.testing.assert_array_equal(first.onsets, second.onsets)


def test_sidechained_layers_of_a_drop_share_one_sidechain() -> None:
    every_layer = [
        layer.model_copy(update={"track_count_weights": {2: 1.0}})
        for layer in CONFIG.synth.layers
    ]
    config = CONFIG.synth.model_copy(
        update={
            "layers": every_layer,
            "master": None,
            "sidechain_probability": 1.0,
        }
    )
    sidechains = []
    for seed in range(3):
        with patch(
            "src.services.synthesis.synth.Sidechain", wraps=Sidechain
        ) as sidechain:
            generate_drop(_bank(), np.random.default_rng(seed), config)
        sidechain.assert_called_once()
        sidechains.append(sidechain.call_args.kwargs)

    for drawn in sidechains:
        low, high = config.sidechain_ratio_range
        assert low <= drawn["ratio"] <= high
        low, high = config.sidechain_seconds_range
        assert low <= drawn["duration"] <= high
    assert len({drawn["ratio"] for drawn in sidechains}) == 3
    assert len({drawn["duration"] for drawn in sidechains}) == 3


def test_drop_without_pumping_has_no_sidechain() -> None:
    config = CONFIG.synth.model_copy(
        update={"master": None, "sidechain_probability": 0.0}
    )
    with patch(
        "src.services.synthesis.synth.Sidechain", wraps=Sidechain
    ) as sidechain:
        generate_drop(_bank(), np.random.default_rng(0), config)

    assert sidechain.call_args.kwargs["ratio"] == 0.0
