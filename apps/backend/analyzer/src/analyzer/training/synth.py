"""Synthetic rawstyle drops from the sample bank, with exact kick onsets.

A drop has a tempo of 150-200 BPM (160 BPM in half of the drops), one kick
design (a new key every two bars), a kick pattern with beats, off-beats, rolls,
triplets and gaps, up to three ducked loop layers, and one-shot hits. The kicks
and one-shots keep their sound at every tempo; only the loops (160 BPM in the
bank) are resampled to the drop tempo. The mix is mastered with EQ, soft
clipping and a limiter. Each kick cuts the tail of the one before it.

The chances, ranges and levels are in `SynthSettings` (analyzer.settings).
"""

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pyloudnorm
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt

from analyzer.audio.io import SAMPLE_RATE, TARGET_LUFS, db_to_gain
from analyzer.settings import (
    KickSettings,
    LoopSettings,
    MasterSettings,
    SynthSettings,
)
from analyzer.training.bank import BPM, Bank, is_held_out

BEATS_PER_BAR = 4
BARS_PER_PHRASE = 4
TICKS_PER_BEAT = 12  # 1/16 notes are 3 ticks and triplets 4 ticks
KICK_LEVEL_SECONDS = 60.0 / BPM  # the kick level is measured on its first beat
HIGH_PASS_ORDER = 2  # 12 dB per octave

_meter = pyloudnorm.Meter(SAMPLE_RATE)


@dataclass
class Drop:
    audio: np.ndarray  # mono float32 at SAMPLE_RATE
    onsets: np.ndarray  # kick onset times in seconds
    info: dict[str, Any] = field(default_factory=dict)  # the samples and settings


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


def compute_rms(signal: np.ndarray) -> float:
    """The root mean square level of a signal."""
    return float(np.sqrt(np.mean(np.square(signal)) + 1e-12))


def compute_attack_rms(sound: np.ndarray) -> float:
    """The RMS level of the first quarter of a one-shot (its attack and body)."""
    return compute_rms(sound[: len(sound) // 4 + 1])


def mix_into(out: np.ndarray, sound: np.ndarray, start: int, gain: float) -> None:
    """Add `sound` into `out` at sample `start` (which can be negative)."""
    begin, end = max(start, 0), min(start + len(sound), len(out))
    if begin < end:
        out[begin:end] += gain * sound[begin - start : end - start]


def draw_from(rng: np.random.Generator, distribution: dict[int, float]) -> int:
    """A key of `distribution`, drawn with the probabilities in its values."""
    keys = list(distribution)
    return keys[rng.choice(len(keys), p=list(distribution.values()))]


def draw_drop_tempo(rng: np.random.Generator, settings: SynthSettings) -> int:
    """A whole BPM from one of `settings.tempos`."""
    bands = settings.tempos
    band = bands[rng.choice(len(bands), p=[b.chance for b in bands])]
    return int(rng.integers(band.low_bpm, band.high_bpm + 1))


def draw_kick_pattern(
    rng: np.random.Generator, bars: int, settings: KickSettings
) -> list[tuple[int, bool]]:
    """Kick positions in ticks from the first beat, and whether each is part of a roll."""
    kicks: list[tuple[int, bool]] = []
    if rng.random() < settings.kickless_drop_chance:
        return kicks
    off_beat = rng.uniform(*settings.off_beat_rate)
    missing = rng.uniform(*settings.missing_rate)
    for bar in range(bars):
        if rng.random() < settings.kickless_bar_chance:
            continue
        phrase_end = bar % BARS_PER_PHRASE == BARS_PER_PHRASE - 1
        roll_chance = (
            settings.phrase_end_roll_chance if phrase_end else settings.roll_chance
        )
        roll_beats = (
            draw_from(rng, settings.roll_beats) if rng.random() < roll_chance else 0
        )
        for beat in range(BEATS_PER_BAR):
            tick = (BEATS_PER_BAR * bar + beat) * TICKS_PER_BEAT
            if beat >= BEATS_PER_BAR - roll_beats:
                step = draw_from(rng, settings.roll_steps)
                start = step if rng.random() < settings.late_roll_chance else 0
                kicks += [(tick + t, True) for t in range(start, TICKS_PER_BEAT, step)]
                continue
            if rng.random() >= missing:
                kicks.append((tick, False))
            if rng.random() < off_beat:
                kicks.append((tick + TICKS_PER_BEAT // 2, False))
            elif rng.random() < settings.syncopated_chance:
                kicks.append((tick + int(rng.choice(settings.syncopated_ticks)), False))
    return kicks


def render_kicks(
    rng: np.random.Generator,
    bank: Bank,
    files: list[int],
    starts: np.ndarray,
    in_roll: np.ndarray,
    length: int,
    bar: float,
    settings: KickSettings,
) -> tuple[np.ndarray, list[int]]:
    """The kick bus (each kick plays until the next kick starts) and the files used.

    `starts` and `bar` are in samples.
    """
    out = np.zeros(length, dtype=np.float32)
    used: list[int] = []
    key, key_section = int(rng.choice(files)), -1
    fade = int(settings.cut_fade * SAMPLE_RATE)
    for i, start in enumerate(starts):
        section = int(max(start, 0) // (settings.key_change_bars * bar))
        if section != key_section:
            key_section = section
            if rng.random() < settings.key_change_chance:
                key = int(rng.choice(files))
        kick = bank.get_audio(key)
        if i + 1 < len(starts) and starts[i + 1] - start < len(kick):
            kick = kick[: max(int(starts[i + 1] - start), 1)]
            n = min(fade, len(kick))
            kick[len(kick) - n :] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        level = settings.roll_level_db if in_roll[i] else settings.level_db
        mix_into(out, kick, int(start), db_to_gain(rng.uniform(*level)))
        if key not in used:
            used.append(key)
    return out, used


def make_sidechain_curve(
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


def high_pass(signal: np.ndarray, cutoff_hz: float) -> np.ndarray:
    sos = butter(HIGH_PASS_ORDER, cutoff_hz, "highpass", fs=SAMPLE_RATE, output="sos")
    return cast(np.ndarray, sosfilt(sos, signal)).astype(np.float32)


def render_loop(
    rng: np.random.Generator,
    loop: np.ndarray,
    origin: int,
    length: int,
    bar: float,
    settings: LoopSettings,
) -> np.ndarray:
    """A loop tiled over the drop, on bars of `bar` samples (sometimes off the bar)."""
    period = int(max(1, round(len(loop) / bar)) * bar)
    tile = np.zeros(period, dtype=np.float32)
    tile[: min(period, len(loop))] = loop[:period]
    shift = int(rng.integers(0, period // int(bar))) * int(bar)
    if rng.random() < settings.off_bar_chance:
        shift = int(rng.integers(0, period))
    n = length - origin + shift
    out = np.tile(tile, n // period + 1)[shift - origin : shift - origin + length]
    return out[:length]


def master_mix(
    rng: np.random.Generator, mix: np.ndarray, settings: MasterSettings
) -> tuple[np.ndarray, dict]:
    """EQ tilt, drive into a soft clipper, and a peak limiter; also the settings used."""
    used: dict[str, Any] = {"eq_tilt": None}
    if rng.random() < settings.eq_tilt_chance:
        corner = rng.uniform(*settings.eq_corner_hz)
        b, a = cast(
            tuple[np.ndarray, np.ndarray],
            butter(1, corner, fs=SAMPLE_RATE, output="ba"),
        )  # first-order low band
        low = cast(np.ndarray, lfilter(b, a, mix))
        low_db, high_db = rng.uniform(*settings.eq_gain_db, size=2)
        mix = low * db_to_gain(low_db) + (mix - low) * db_to_gain(high_db)
        used["eq_tilt"] = {
            "corner_hz": round(corner),
            "low_db": round(low_db, 1),
            "high_db": round(high_db, 1),
        }
    drive = rng.uniform(*settings.drive_db)
    mix = scale_to_lufs(mix) * db_to_gain(drive)
    used["drive_db"] = round(drive, 1)
    used["clipper"] = rng.random() < settings.clipper_chance
    if used["clipper"]:
        mix = np.tanh(mix)
    used["limiter"] = rng.random() < settings.limiter_chance
    if used["limiter"]:
        window = int(settings.limiter_window * SAMPLE_RATE)
        envelope = maximum_filter1d(np.abs(mix), window)
        gain = uniform_filter1d(
            np.minimum(1.0, settings.limiter_ceiling / np.maximum(envelope, 1e-9)),
            window,
        )
        mix = mix * gain
    return np.clip(mix, -1.0, 1.0), used


def scale_to_lufs(signal: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """`signal` scaled to an integrated loudness of `target_lufs` (unchanged if silent)."""
    loudness = _meter.integrated_loudness(signal)
    if not np.isfinite(loudness):
        return signal
    return signal * db_to_gain(target_lufs - loudness)


def make_drop(
    rng: np.random.Generator,
    catalog: Catalog,
    settings: SynthSettings,
    held_out: bool = False,
) -> Drop:
    """One synthetic drop; `held_out` uses only held-out kick designs."""
    bank = catalog.bank
    bpm = draw_drop_tempo(rng, settings)
    beat = 60.0 / bpm * SAMPLE_RATE  # in samples
    bar = BEATS_PER_BAR * beat
    length = int(settings.drop_seconds * SAMPLE_RATE)
    origin = -int(rng.uniform(0.0, bar))  # first beat of bar 1
    bars = int(np.ceil((length - origin) / bar))
    tick = beat / TICKS_PER_BEAT

    pattern = draw_kick_pattern(rng, bars, settings.kicks)
    starts = np.array([origin + round(t * tick) for t, _ in pattern], dtype=float)
    in_roll = np.array([r for _, r in pattern], dtype=bool)
    files = catalog.designs[held_out][int(rng.integers(len(catalog.designs[held_out])))]
    kicks, used = render_kicks(
        rng, bank, files, starts, in_roll, length, bar, settings.kicks
    )
    first_kick = bank.get_audio(files[0])
    reference = compute_rms(first_kick[: int(KICK_LEVEL_SECONDS * SAMPLE_RATE)])

    def to_seconds(position: float) -> float:
        return round(position / SAMPLE_RATE, 3)

    path = [s.path for s in bank.samples]
    info: dict[str, Any] = {
        "held_out": held_out,
        "bpm": bpm,
        "kick_design": bank.samples[files[0]].design,
        "kick_files": [path[i] for i in used],
        "loops": [],
        "clap": None,
        "hits": [],
        "impact": None,
    }

    mix = kicks.copy()
    loops = settings.loops
    ducking = make_sidechain_curve(rng, length, starts, loops)
    layers = draw_from(rng, loops.layers)
    for _ in range(layers if len(catalog.loops) else 0):
        index = int(rng.choice(catalog.loops))
        loop = bank.get_audio(index)
        if bpm != BPM:  # the bank loops are at BPM
            loop = resample_poly(loop, int(BPM), bpm).astype(np.float32)
        loop = render_loop(rng, loop, origin, length, bar, loops)
        cutoff = None
        if rng.random() < loops.high_pass_chance:
            cutoff = rng.uniform(*loops.high_pass_hz)
            loop = high_pass(loop, cutoff)
        level = rng.uniform(*(loops.level_db if cutoff else loops.full_band_level_db))
        gain = reference * db_to_gain(level) / compute_rms(loop)
        ducked = rng.random() < loops.duck_chance
        mix += gain * loop * (ducking if ducked else 1.0)
        info["loops"].append(
            {
                "path": path[index],
                "high_pass_hz": round(cutoff) if cutoff else None,
                "level_db": round(level, 1),  # relative to the kick
                "ducked": ducked,
            }
        )

    claps = settings.claps
    if len(catalog.claps) and rng.random() < claps.chance:
        index = int(rng.choice(catalog.claps))
        clap = bank.get_audio(index)
        level = rng.uniform(*claps.level_db)
        gain = db_to_gain(level) * reference / compute_attack_rms(clap)
        for backbeat in range(1, BEATS_PER_BAR * bars, 2):  # beats 2 and 4
            if rng.random() < claps.beat_chance:
                mix_into(mix, clap, origin + round(backbeat * beat), gain)
        info["clap"] = {"path": path[index], "level_db": round(level, 1)}
    hits = settings.hits
    sixteenth = beat / 4
    for _ in range(int(rng.poisson(hits.mean_count)) if len(catalog.hits) else 0):
        index = int(rng.choice(catalog.hits))
        hit = bank.get_audio(index)
        at = int(rng.integers(0, BEATS_PER_BAR * 4 * bars)) * sixteenth
        if rng.random() < hits.off_grid_chance:
            at += rng.uniform(*hits.off_grid_shift) * SAMPLE_RATE
        level = rng.uniform(*hits.level_db)
        gain = db_to_gain(level) * reference / compute_attack_rms(hit)
        mix_into(mix, hit, origin + round(at), gain)
        info["hits"].append(
            {
                "path": path[index],
                "time": to_seconds(origin + round(at)),
                "level_db": round(level, 1),
            }
        )
    info["hits"].sort(key=lambda hit: hit["time"])
    impacts = settings.impacts
    if len(catalog.impacts) and rng.random() < impacts.chance:
        index = int(rng.choice(catalog.impacts))
        impact = bank.get_audio(index)
        level = rng.uniform(*impacts.level_db)
        gain = db_to_gain(level) * reference / compute_rms(impact)
        at = origin + round(bar * rng.choice(impacts.bars))
        mix_into(mix, impact, at, gain)
        info["impact"] = {
            "path": path[index],
            "time": to_seconds(at),
            "level_db": round(level, 1),
        }

    mix, info["master"] = master_mix(rng, mix, settings.master)
    gain_db = rng.uniform(*settings.master.output_gain_db)
    mix = scale_to_lufs(mix) * db_to_gain(gain_db)
    info["gain_db"] = round(gain_db, 1)
    onsets = starts / SAMPLE_RATE
    onsets = onsets[(onsets >= 0.0) & (onsets < settings.drop_seconds)]
    info["kicks"] = len(onsets)
    return Drop(mix.astype(np.float32), onsets, info)
