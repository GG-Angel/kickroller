"""Synthetic rawstyle drops from the sample bank, with exact kick onsets.

A drop has a tempo of 150-200 BPM (160 BPM in half of the drops), one kick
design (a new key every two bars), a kick pattern with beats, off-beats, rolls,
triplets and gaps, up to three ducked loop layers, and one-shot hits. The kicks
and one-shots keep their sound at every tempo; only the loops (160 BPM in the
bank) are resampled to the drop tempo. The mix is mastered with EQ, soft
clipping and a limiter. Each kick cuts the tail of the one before it.

`make_drop` first draws the grid and the kicks. Then each `_render_*` function
makes one track (kicks, loops, claps, hits, impact) and its info (the samples
and settings used). The levels of the other tracks are relative to the kick
level. Last, the sum of the tracks is mastered.

The chances, ranges and levels are in `SynthSettings` (analyzer.settings).
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import numpy as np
import pyloudnorm
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt

from analyzer.audio.io import SAMPLE_RATE, TARGET_LUFS, db_to_gain
from analyzer.settings import (
    ClapSettings,
    HitSettings,
    ImpactSettings,
    KickSettings,
    LoopSettings,
    MasterSettings,
    SynthSettings,
)
from analyzer.training.bank import BPM, Bank, is_held_out

BEATS_PER_BAR = 4
BARS_PER_PHRASE = 4
TICKS_PER_BEAT = 12  # 1/16 notes are 3 ticks and triplets 4 ticks
SIXTEENTHS_PER_BEAT = 4
KICK_LEVEL_SECONDS = 60.0 / BPM  # the kick level is measured on its first beat
HIGH_PASS_ORDER = 2  # 12 dB per octave
EQ_TILT_ORDER = 1  # the low band filter of the EQ tilt (6 dB per octave)

_meter = pyloudnorm.Meter(SAMPLE_RATE)

_Info = dict[str, Any]  # the samples and settings of one track, for `Drop.info`


@dataclass
class Drop:
    audio: np.ndarray  # mono float32 at SAMPLE_RATE
    onsets: np.ndarray  # kick onset times in seconds
    info: _Info = field(default_factory=dict)  # the samples and settings


class Catalog:
    """The bank samples by role, split into training and held-out kick designs.

    About `held_out_fraction` of the designs are held out. With `all_designs`
    (for a final model), training also uses the held-out designs.
    """

    def __init__(
        self, bank: Bank, held_out_fraction: float, all_designs: bool = False
    ) -> None:
        self.bank = bank
        designs: dict[str, list[int]] = {}
        for i in bank.get_indices("kick"):
            designs.setdefault(bank.samples[i].design, []).append(int(i))
        self.designs = {
            held_out: [
                files
                for name, files in sorted(designs.items())
                if is_held_out(name, held_out_fraction) == held_out
            ]
            for held_out in (False, True)
        }
        if all_designs:
            self.designs[False] = [files for _, files in sorted(designs.items())]
        self.loops = bank.get_indices("loop")
        self.claps = bank.get_indices("clap")
        self.impacts = bank.get_indices("impact")
        self.hits = bank.get_indices("hit")


@dataclass(frozen=True)
class _Grid:
    """The time grid of a drop. All positions are in samples from the drop start."""

    bpm: int
    beat: float  # the length of one beat
    origin: int  # the first beat of bar 1 (at or before the drop start)
    length: int  # the length of the drop
    bars: int  # the bars from `origin` to the drop end

    @property
    def bar(self) -> float:
        return BEATS_PER_BAR * self.beat

    @property
    def tick(self) -> float:
        return self.beat / TICKS_PER_BEAT


class _Kicks(NamedTuple):
    starts: np.ndarray  # in samples from the drop start (whole numbers)
    in_roll: np.ndarray  # whether each kick is part of a roll


# Levels and mixing


def _compute_rms(signal: np.ndarray) -> float:
    """The root mean square level of a signal."""
    return float(np.sqrt(np.mean(np.square(signal)) + 1e-12))


def _compute_attack_rms(sound: np.ndarray) -> float:
    """The RMS level of the first quarter of a one-shot (its attack and body)."""
    return _compute_rms(sound[: len(sound) // 4 + 1])


def _relative_gain(level_db: float, reference: float, level: float) -> float:
    """The gain that puts a sound with RMS `level` at `level_db` relative to `reference`."""
    return db_to_gain(level_db) * reference / level


def _scale_to_lufs(signal: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """`signal` scaled to an integrated loudness of `target_lufs` (unchanged if silent)."""
    loudness = _meter.integrated_loudness(signal)
    if not np.isfinite(loudness):
        return signal
    return signal * db_to_gain(target_lufs - loudness)


def _mix_into(out: np.ndarray, sound: np.ndarray, start: int, gain: float) -> None:
    """Add `sound` into `out` at sample `start` (which can be negative)."""
    begin, end = max(start, 0), min(start + len(sound), len(out))
    if begin < end:
        out[begin:end] += gain * sound[begin - start : end - start]


def _to_seconds(position: float) -> float:
    """A position in samples as seconds, rounded to 1 ms (for `Drop.info`)."""
    return round(position / SAMPLE_RATE, 3)


# Random choices


def _draw_from(rng: np.random.Generator, distribution: dict[int, float]) -> int:
    """A key of `distribution`, drawn with the probabilities in its values."""
    keys = list(distribution)
    return keys[rng.choice(len(keys), p=list(distribution.values()))]


def _draw_sample(
    rng: np.random.Generator, bank: Bank, indices: np.ndarray
) -> tuple[str, np.ndarray]:
    """The path and the audio of a random sample from `indices`."""
    index = int(rng.choice(indices))
    return bank.samples[index].path, bank.get_audio(index)


def _draw_drop_tempo(rng: np.random.Generator, settings: SynthSettings) -> int:
    """A whole BPM from one of `settings.tempos`."""
    bands = settings.tempos
    band = bands[rng.choice(len(bands), p=[b.chance for b in bands])]
    return int(rng.integers(band.low_bpm, band.high_bpm + 1))


def _draw_grid(rng: np.random.Generator, settings: SynthSettings) -> _Grid:
    """A drop tempo and phase: bar 1 starts up to one bar before the drop."""
    bpm = _draw_drop_tempo(rng, settings)
    beat = 60.0 / bpm * SAMPLE_RATE
    length = int(settings.drop_seconds * SAMPLE_RATE)
    origin = -int(rng.uniform(0.0, BEATS_PER_BAR * beat))
    bars = int(np.ceil((length - origin) / (BEATS_PER_BAR * beat)))
    return _Grid(bpm, beat, origin, length, bars)


# Kicks


def _draw_roll_beats(rng: np.random.Generator, bar: int, settings: KickSettings) -> int:
    """The number of beats at the end of `bar` that are a kick roll (0: no roll)."""
    phrase_end = bar % BARS_PER_PHRASE == BARS_PER_PHRASE - 1
    chance = settings.phrase_end_roll_chance if phrase_end else settings.roll_chance
    return _draw_from(rng, settings.roll_beats) if rng.random() < chance else 0


def _draw_roll(
    rng: np.random.Generator, tick: int, settings: KickSettings
) -> list[int]:
    """The kicks of one roll beat from `tick`, one each `roll_steps` ticks."""
    step = _draw_from(rng, settings.roll_steps)
    start = step if rng.random() < settings.late_roll_chance else 0
    return [tick + t for t in range(start, TICKS_PER_BEAT, step)]


def _draw_beat(
    rng: np.random.Generator,
    tick: int,
    missing_rate: float,
    off_beat_rate: float,
    settings: KickSettings,
) -> list[int]:
    """The kicks of one beat from `tick`: on the beat, then a 1/8 off-beat or a syncopated kick."""
    kicks = []
    if rng.random() >= missing_rate:
        kicks.append(tick)
    if rng.random() < off_beat_rate:
        kicks.append(tick + TICKS_PER_BEAT // 2)
    elif rng.random() < settings.syncopated_chance:
        kicks.append(tick + int(rng.choice(settings.syncopated_ticks)))
    return kicks


def _draw_kick_pattern(
    rng: np.random.Generator, bars: int, settings: KickSettings
) -> list[tuple[int, bool]]:
    """Kick positions in ticks from the first beat, and whether each is part of a roll."""
    if rng.random() < settings.kickless_drop_chance:
        return []
    off_beat_rate = rng.uniform(*settings.off_beat_rate)
    missing_rate = rng.uniform(*settings.missing_rate)
    kicks: list[tuple[int, bool]] = []
    for bar in range(bars):
        if rng.random() < settings.kickless_bar_chance:
            continue
        roll_beats = _draw_roll_beats(rng, bar, settings)
        for beat in range(BEATS_PER_BAR):
            tick = (BEATS_PER_BAR * bar + beat) * TICKS_PER_BEAT
            if beat >= BEATS_PER_BAR - roll_beats:
                kicks += [(t, True) for t in _draw_roll(rng, tick, settings)]
            else:
                beat_kicks = _draw_beat(
                    rng, tick, missing_rate, off_beat_rate, settings
                )
                kicks += [(t, False) for t in beat_kicks]
    return kicks


def _draw_kicks(
    rng: np.random.Generator, grid: _Grid, settings: KickSettings
) -> _Kicks:
    """The kick pattern of a drop, placed on its grid."""
    pattern = _draw_kick_pattern(rng, grid.bars, settings)
    starts = [grid.origin + round(tick * grid.tick) for tick, _ in pattern]
    in_roll = [roll for _, roll in pattern]
    return _Kicks(np.array(starts, dtype=float), np.array(in_roll, dtype=bool))


def _draw_design(
    rng: np.random.Generator, catalog: Catalog, held_out: bool
) -> list[int]:
    """The bank indices of all versions (keys) of a random kick design."""
    designs = catalog.designs[held_out]
    return designs[int(rng.integers(len(designs)))]


def _draw_kick_sounds(
    rng: np.random.Generator,
    design: list[int],
    kicks: _Kicks,
    grid: _Grid,
    settings: KickSettings,
) -> tuple[list[int], list[float]]:
    """The bank index and the level in dB of each kick.

    The key can change every `key_change_bars` bars. Roll kicks can be quieter.
    """
    keys: list[int] = []
    levels: list[float] = []
    key, key_section = int(rng.choice(design)), -1
    section_length = settings.key_change_bars * grid.bar
    for start, in_roll in zip(kicks.starts, kicks.in_roll, strict=True):
        section = int(max(start, 0) // section_length)
        if section != key_section:
            key_section = section
            if rng.random() < settings.key_change_chance:
                key = int(rng.choice(design))
        keys.append(key)
        level = settings.roll_level_db if in_roll else settings.level_db
        levels.append(rng.uniform(*level))
    return keys, levels


def _measure_kick_level(bank: Bank, design: list[int]) -> float:
    """The RMS level of the first beat of a kick design (the reference level)."""
    kick = bank.get_audio(design[0])
    return _compute_rms(kick[: int(KICK_LEVEL_SECONDS * SAMPLE_RATE)])


def _cut_tail(sound: np.ndarray, room: float, fade: int) -> np.ndarray:
    """`sound` cut to `room` samples (at least 1) with a linear fade-out of `fade` samples.

    The sound does not change if it fits in `room` (which can be infinite). The
    fade changes `sound` in place.
    """
    if room >= len(sound):
        return sound
    sound = sound[: max(int(room), 1)]
    n = min(fade, len(sound))
    sound[len(sound) - n :] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return sound


def _render_kicks(
    bank: Bank,
    kicks: _Kicks,
    keys: list[int],
    levels: list[float],
    grid: _Grid,
    settings: KickSettings,
) -> np.ndarray:
    """The kick track: each kick plays until the next kick starts."""
    out = np.zeros(grid.length, dtype=np.float32)
    fade = int(settings.cut_fade * SAMPLE_RATE)
    rooms = np.diff(kicks.starts, append=np.inf)  # the last kick is not cut
    for start, room, key, level in zip(kicks.starts, rooms, keys, levels, strict=True):
        kick = _cut_tail(bank.get_audio(key), room, fade)
        _mix_into(out, kick, int(start), db_to_gain(level))
    return out


# Loops


def _make_sidechain_curve(
    rng: np.random.Generator, length: int, starts: np.ndarray, settings: LoopSettings
) -> np.ndarray:
    """A gain curve that dips at each kick start (in samples) and recovers."""
    depth = rng.uniform(*settings.duck_depth)
    release = rng.uniform(*settings.duck_release) * SAMPLE_RATE
    impulses = np.zeros(length)
    starts = starts[(starts >= 0) & (starts < length)].astype(int)
    impulses[starts] = 1.0
    curve = lfilter([1.0], [1.0, -np.exp(-1.0 / release)], impulses)
    return (1.0 - depth * np.minimum(curve, 1.0)).astype(np.float32)


def _high_pass(signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
    sos = butter(HIGH_PASS_ORDER, cutoff_hz, "highpass", fs=SAMPLE_RATE, output="sos")
    return cast(np.ndarray, sosfilt(sos, signal)).astype(np.float32)


def _fit_loop_tempo(loop: np.ndarray, bpm: int) -> np.ndarray:
    """A bank loop (at BPM) resampled to `bpm`; this also changes its pitch."""
    if bpm == BPM:
        return loop
    return resample_poly(loop, int(BPM), bpm).astype(np.float32)


def _tile_loop(
    rng: np.random.Generator, loop: np.ndarray, grid: _Grid, settings: LoopSettings
) -> np.ndarray:
    """`loop` repeated over the drop, from a random bar of the loop.

    The loop is padded or cut to whole bars, and its bars are on the drop bars
    (except with `off_bar_chance`: then it starts at a random sample).
    """
    period = int(max(1, round(len(loop) / grid.bar)) * grid.bar)
    tile = np.zeros(period, dtype=np.float32)
    tile[: min(period, len(loop))] = loop[:period]
    bar = int(grid.bar)
    shift = int(rng.integers(0, period // bar)) * bar
    if rng.random() < settings.off_bar_chance:
        shift = int(rng.integers(0, period))
    start = shift - grid.origin  # the position in the tile at the drop start
    repeats = (start + grid.length) // period + 1
    return np.tile(tile, repeats)[start : start + grid.length]


def _render_loop_layer(
    rng: np.random.Generator,
    catalog: Catalog,
    grid: _Grid,
    ducking: np.ndarray,
    reference: float,
    settings: LoopSettings,
) -> tuple[np.ndarray, _Info]:
    """One random loop over the drop, usually high-passed and ducked."""
    path, loop = _draw_sample(rng, catalog.bank, catalog.loops)
    loop = _tile_loop(rng, _fit_loop_tempo(loop, grid.bpm), grid, settings)
    cutoff = None
    if rng.random() < settings.high_pass_chance:
        cutoff = rng.uniform(*settings.high_pass_hz)
        loop = _high_pass(loop, cutoff)
    level_range = (
        settings.level_db if cutoff is not None else settings.full_band_level_db
    )
    level = rng.uniform(*level_range)
    ducked = rng.random() < settings.duck_chance
    gain = _relative_gain(level, reference, _compute_rms(loop))
    info = {
        "path": path,
        "high_pass_hz": round(cutoff) if cutoff is not None else None,
        "level_db": round(level, 1),  # relative to the kick
        "ducked": ducked,
    }
    return gain * loop * (ducking if ducked else 1.0), info


def _render_loops(
    rng: np.random.Generator,
    catalog: Catalog,
    grid: _Grid,
    kicks: _Kicks,
    reference: float,
    settings: LoopSettings,
) -> tuple[np.ndarray, list[_Info]]:
    """The loop track: a random number of loop layers with one sidechain curve."""
    out = np.zeros(grid.length, dtype=np.float32)
    ducking = _make_sidechain_curve(rng, grid.length, kicks.starts, settings)
    layers = _draw_from(rng, settings.layers)
    infos = []
    for _ in range(layers if len(catalog.loops) else 0):
        layer, info = _render_loop_layer(
            rng, catalog, grid, ducking, reference, settings
        )
        out += layer
        infos.append(info)
    return out, infos


# One-shots


def _render_claps(
    rng: np.random.Generator,
    catalog: Catalog,
    grid: _Grid,
    reference: float,
    settings: ClapSettings,
) -> tuple[np.ndarray, _Info | None]:
    """The clap track (in some drops): one clap sample on most beats 2 and 4."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not len(catalog.claps) or rng.random() >= settings.chance:
        return out, None
    path, clap = _draw_sample(rng, catalog.bank, catalog.claps)
    level = rng.uniform(*settings.level_db)
    gain = _relative_gain(level, reference, _compute_attack_rms(clap))
    for backbeat in range(1, BEATS_PER_BAR * grid.bars, 2):  # beats 2 and 4
        if rng.random() < settings.beat_chance:
            _mix_into(out, clap, grid.origin + round(backbeat * grid.beat), gain)
    return out, {"path": path, "level_db": round(level, 1)}


def _render_hits(
    rng: np.random.Generator,
    catalog: Catalog,
    grid: _Grid,
    reference: float,
    settings: HitSettings,
) -> tuple[np.ndarray, list[_Info]]:
    """The hit track: random one-shots on the 1/16 grid (some off it).

    Each hit starts in the drop. The info is in time order.
    """
    out = np.zeros(grid.length, dtype=np.float32)
    if not len(catalog.hits):
        return out, []
    sixteenth = grid.beat / SIXTEENTHS_PER_BEAT
    # The 1/16 notes from bar 1 that start in the drop: from `first` to before `end`.
    first = int(np.ceil(-grid.origin / sixteenth))
    end = int(np.ceil((grid.length - grid.origin) / sixteenth))
    infos = []
    for _ in range(int(rng.poisson(settings.mean_count))):
        path, hit = _draw_sample(rng, catalog.bank, catalog.hits)
        on_grid = int(rng.integers(first, end)) * sixteenth
        shift = 0.0
        if rng.random() < settings.off_grid_chance:
            shift = rng.uniform(*settings.off_grid_shift) * SAMPLE_RATE
        level = rng.uniform(*settings.level_db)
        start = grid.origin + round(on_grid + shift)
        if not 0 <= start < grid.length:  # the shift moved the hit out of the drop
            start = grid.origin + round(on_grid)
        gain = _relative_gain(level, reference, _compute_attack_rms(hit))
        _mix_into(out, hit, start, gain)
        infos.append(
            {"path": path, "time": _to_seconds(start), "level_db": round(level, 1)}
        )
    return out, sorted(infos, key=lambda info: info["time"])


def _render_impact(
    rng: np.random.Generator,
    catalog: Catalog,
    grid: _Grid,
    reference: float,
    settings: ImpactSettings,
) -> tuple[np.ndarray, _Info | None]:
    """The impact track (in some drops): one impact at the start of one of `settings.bars`."""
    out = np.zeros(grid.length, dtype=np.float32)
    if not len(catalog.impacts) or rng.random() >= settings.chance:
        return out, None
    path, impact = _draw_sample(rng, catalog.bank, catalog.impacts)
    level = rng.uniform(*settings.level_db)
    start = grid.origin + round(grid.bar * rng.choice(settings.bars))
    _mix_into(
        out, impact, start, _relative_gain(level, reference, _compute_rms(impact))
    )
    return out, {"path": path, "time": _to_seconds(start), "level_db": round(level, 1)}


# Mastering


def _tilt_eq(
    rng: np.random.Generator, mix: np.ndarray, settings: MasterSettings
) -> tuple[np.ndarray, _Info | None]:
    """In some drops, a low and a high band, each with a random gain."""
    if rng.random() >= settings.eq_tilt_chance:
        return mix, None
    corner = rng.uniform(*settings.eq_corner_hz)
    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(EQ_TILT_ORDER, corner, fs=SAMPLE_RATE, output="ba"),
    )
    low = cast(np.ndarray, lfilter(b, a, mix))
    low_db, high_db = rng.uniform(*settings.eq_gain_db, size=2)
    mix = low * db_to_gain(low_db) + (mix - low) * db_to_gain(high_db)
    info = {
        "corner_hz": round(corner),
        "low_db": round(low_db, 1),
        "high_db": round(high_db, 1),
    }
    return mix, info


def _limit_peaks(mix: np.ndarray, settings: MasterSettings) -> np.ndarray:
    """A peak limiter: a smoothed gain that keeps the peaks near `limiter_ceiling`."""
    window = int(settings.limiter_window * SAMPLE_RATE)
    envelope = maximum_filter1d(np.abs(mix), window)
    gain = np.minimum(1.0, settings.limiter_ceiling / np.maximum(envelope, 1e-9))
    return mix * uniform_filter1d(gain, window)


def _master_mix(
    rng: np.random.Generator, mix: np.ndarray, settings: MasterSettings
) -> tuple[np.ndarray, _Info]:
    """EQ tilt, drive into a soft clipper, and a peak limiter; also the settings used."""
    mix, eq_tilt = _tilt_eq(rng, mix, settings)
    drive = rng.uniform(*settings.drive_db)
    mix = _scale_to_lufs(mix) * db_to_gain(drive)
    clipper = rng.random() < settings.clipper_chance
    if clipper:
        mix = np.tanh(mix)
    limiter = rng.random() < settings.limiter_chance
    if limiter:
        mix = _limit_peaks(mix, settings)
    info = {
        "eq_tilt": eq_tilt,
        "drive_db": round(drive, 1),
        "clipper": clipper,
        "limiter": limiter,
    }
    return np.clip(mix, -1.0, 1.0), info


def _set_output_level(
    rng: np.random.Generator, mix: np.ndarray, settings: MasterSettings
) -> tuple[np.ndarray, float]:
    """The mix at TARGET_LUFS plus a random gain in dB (tracks differ in loudness)."""
    gain_db = rng.uniform(*settings.output_gain_db)
    return _scale_to_lufs(mix) * db_to_gain(gain_db), gain_db


def make_drop(
    rng: np.random.Generator,
    catalog: Catalog,
    settings: SynthSettings,
    held_out: bool = False,
) -> Drop:
    """One synthetic drop; `held_out` uses only held-out kick designs."""
    bank = catalog.bank
    grid = _draw_grid(rng, settings)
    kicks = _draw_kicks(rng, grid, settings.kicks)
    design = _draw_design(rng, catalog, held_out)
    keys, levels = _draw_kick_sounds(rng, design, kicks, grid, settings.kicks)
    reference = _measure_kick_level(bank, design)

    kick_track = _render_kicks(bank, kicks, keys, levels, grid, settings.kicks)
    loop_track, loops = _render_loops(
        rng, catalog, grid, kicks, reference, settings.loops
    )
    clap_track, clap = _render_claps(rng, catalog, grid, reference, settings.claps)
    hit_track, hits = _render_hits(rng, catalog, grid, reference, settings.hits)
    impact_track, impact = _render_impact(
        rng, catalog, grid, reference, settings.impacts
    )
    mix = kick_track + loop_track + clap_track + hit_track + impact_track
    mix, master = _master_mix(rng, mix, settings.master)
    mix, gain_db = _set_output_level(rng, mix, settings.master)

    in_drop = (kicks.starts >= 0) & (kicks.starts < grid.length)
    onsets = kicks.starts[in_drop] / SAMPLE_RATE
    info = {
        "held_out": held_out,
        "bpm": grid.bpm,
        "kick_design": bank.samples[design[0]].design,
        "kick_files": [bank.samples[key].path for key in dict.fromkeys(keys)],
        "loops": loops,
        "clap": clap,
        "hits": hits,
        "impact": impact,
        "master": master,
        "gain_db": round(gain_db, 1),
        "kicks": len(onsets),
    }
    return Drop(mix.astype(np.float32), onsets, info)
