"""Synthetic 160 BPM rawstyle drops from the sample bank, with exact kick onsets.

A drop has one kick design (a new key every two bars), a kick pattern with
beats, off-beats, rolls, triplets and gaps, up to three ducked loop layers, and
one-shot hits. The mix is mastered with EQ, soft clipping and a limiter, then
its tempo changes a little. Each kick cuts the tail of the one before it.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyloudnorm
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter, sosfilt

from analyzer.audio import SAMPLE_RATE, TARGET_LUFS
from analyzer.bank import Bank, is_held_out

BPM = 160.0
BEAT = 60.0 / BPM
TICKS = 12  # grid ticks per beat: 1/16 notes are 3 ticks and triplets 4 ticks
DROP_SECONDS = 12.0  # 8 bars at 160 BPM
HELD_OUT = 0.1  # fraction of the kick designs used only for validation
CUT_FADE = 0.003  # fade-out when the next kick cuts the tail, in seconds

_meter = pyloudnorm.Meter(SAMPLE_RATE)


@dataclass
class Drop:
    audio: np.ndarray  # mono float32 at SAMPLE_RATE
    onsets: np.ndarray  # kick onset times in seconds
    info: dict[str, Any] = field(default_factory=dict)  # the samples and settings


class Catalog:
    """The bank samples by role, split into training and held-out kick designs."""

    def __init__(self, bank: Bank) -> None:
        self.bank = bank
        designs: dict[str, list[int]] = {}
        for i in bank.indices("kick"):
            designs.setdefault(bank.samples[i].design, []).append(int(i))
        self.designs = {
            held_out: [
                files
                for name, files in sorted(designs.items())
                if is_held_out(name, HELD_OUT) == held_out
            ]
            for held_out in (False, True)
        }
        self.loops = bank.indices("loop")
        self.claps = bank.indices("clap")
        self.impacts = bank.indices("impact")
        self.hits = bank.indices("hit")


def db(gain_db: float) -> float:
    return 10.0 ** (gain_db / 20.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def place(out: np.ndarray, sound: np.ndarray, start: int, gain: float = 1.0) -> None:
    """Add `sound` into `out` at sample `start` (which can be negative)."""
    begin, end = max(start, 0), min(start + len(sound), len(out))
    if begin < end:
        out[begin:end] += gain * sound[begin - start : end - start]


def kick_pattern(rng: np.random.Generator, bars: int) -> list[tuple[int, bool]]:
    """Kick positions in ticks from the first beat, and whether each is part of a roll."""
    kicks: list[tuple[int, bool]] = []
    if rng.random() < 0.05:
        return kicks  # a kickless drop
    off_beat = rng.uniform(0.0, 0.3)  # this drop's rate of 1/8 off-beat kicks
    missing = rng.uniform(0.0, 0.15)  # and of missing beat kicks
    for bar in range(bars):
        if rng.random() < 0.05:
            continue  # a kickless bar
        roll_beats = 0
        if rng.random() < (0.4 if bar % 4 == 3 else 0.15):
            roll_beats = int(rng.choice([1, 1, 2, 4]))
        for beat in range(4):
            tick = (4 * bar + beat) * TICKS
            if beat >= 4 - roll_beats:
                step = int(rng.choice([3, 4, 6], p=[0.45, 0.3, 0.25]))
                start = 0 if rng.random() < 0.8 else step
                kicks += [(tick + t, True) for t in range(start, TICKS, step)]
                continue
            if rng.random() >= missing:
                kicks.append((tick, False))
            if rng.random() < off_beat:
                kicks.append((tick + TICKS // 2, False))
            elif rng.random() < 0.03:
                kicks.append((tick + int(rng.choice([3, 4, 8, 9])), False))
    return kicks


def render_kicks(
    rng: np.random.Generator,
    bank: Bank,
    files: list[int],
    starts: np.ndarray,
    in_roll: np.ndarray,
    length: int,
) -> tuple[np.ndarray, list[int]]:
    """The kick bus (each kick plays until the next kick starts) and the files used."""
    out = np.zeros(length, dtype=np.float32)
    used: list[int] = []
    bar_samples = 4 * BEAT * SAMPLE_RATE
    key, key_bar = int(rng.choice(files)), -1
    fade = int(CUT_FADE * SAMPLE_RATE)
    for i, start in enumerate(starts):
        pair = int(max(start, 0) // (2 * bar_samples))
        if pair != key_bar:
            key_bar = pair
            if rng.random() < 0.5:
                key = int(rng.choice(files))
        kick = bank.get(key)
        if i + 1 < len(starts) and starts[i + 1] - start < len(kick):
            kick = kick[: max(int(starts[i + 1] - start), 1)]
            n = min(fade, len(kick))
            kick[len(kick) - n :] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        gain = db(rng.uniform(-6.0, 0.0) if in_roll[i] else rng.uniform(-1.0, 0.0))
        place(out, kick, int(start), gain)
        if key not in used:
            used.append(key)
    return out, used


def duck(length: int, starts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A sidechain gain curve that dips at each kick and recovers."""
    depth = rng.uniform(0.4, 1.0)
    release = rng.uniform(0.05, 0.25) * SAMPLE_RATE
    impulses = np.zeros(length)
    starts = starts[(starts >= 0) & (starts < length)].astype(int)
    impulses[starts] = 1.0
    curve = lfilter([1.0], [1.0, -np.exp(-1.0 / release)], impulses)
    return (1.0 - depth * np.minimum(curve, 1.0)).astype(np.float32)


def high_pass(x: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(2, cutoff, "highpass", fs=SAMPLE_RATE, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def render_loop(
    rng: np.random.Generator, loop: np.ndarray, origin: int, length: int
) -> np.ndarray:
    """A loop tiled over the drop, starting on a bar (sometimes off the bar)."""
    bar = 4 * BEAT * SAMPLE_RATE
    period = int(max(1, round(len(loop) / bar)) * bar)
    tile = np.zeros(period, dtype=np.float32)
    tile[: min(period, len(loop))] = loop[:period]
    shift = int(rng.integers(0, period // int(bar))) * int(bar)
    if rng.random() < 0.2:
        shift = int(rng.integers(0, period))
    n = length - origin + shift
    out = np.tile(tile, n // period + 1)[shift - origin : shift - origin + length]
    return out[:length]


def master(rng: np.random.Generator, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """EQ tilt, drive into a soft clipper, and a peak limiter; also the settings."""
    settings: dict[str, Any] = {"eq_tilt": None}
    if rng.random() < 0.5:
        corner = rng.uniform(300.0, 3000.0)
        b, a = butter(1, corner, fs=SAMPLE_RATE)
        low = lfilter(b, a, x)
        low_db, high_db = rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0)
        x = low * db(low_db) + (x - low) * db(high_db)
        settings["eq_tilt"] = {
            "corner_hz": round(corner),
            "low_db": round(low_db, 1),
            "high_db": round(high_db, 1),
        }
    drive = rng.uniform(0.0, 12.0)
    x = normalize(x) * db(drive)
    settings["drive_db"] = round(drive, 1)
    settings["clipper"] = rng.random() < 0.7
    if settings["clipper"]:
        x = np.tanh(x)
    settings["limiter"] = rng.random() < 0.6
    if settings["limiter"]:
        window = int(0.005 * SAMPLE_RATE)
        envelope = maximum_filter1d(np.abs(x), window)
        gain = uniform_filter1d(
            np.minimum(1.0, 0.95 / np.maximum(envelope, 1e-9)), window
        )
        x = x * gain
    return np.clip(x, -1.0, 1.0), settings


def normalize(x: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    loudness = _meter.integrated_loudness(x)
    if not np.isfinite(loudness):
        return x
    return x * db(target_lufs - loudness)


def make_drop(
    rng: np.random.Generator, catalog: Catalog, held_out: bool = False
) -> Drop:
    """One synthetic drop of DROP_SECONDS; `held_out` uses only held-out kick designs."""
    bank = catalog.bank
    tempo = rng.uniform(0.94, 1.06) if rng.random() < 0.8 else 1.0
    length = int(np.ceil(DROP_SECONDS * tempo * SAMPLE_RATE))
    origin = -int(rng.uniform(0.0, 4 * BEAT) * SAMPLE_RATE)  # first beat of bar 1
    bars = int(np.ceil((length - origin) / (4 * BEAT * SAMPLE_RATE)))
    tick = BEAT * SAMPLE_RATE / TICKS

    pattern = kick_pattern(rng, bars)
    starts = np.array([origin + round(t * tick) for t, _ in pattern], dtype=float)
    in_roll = np.array([r for _, r in pattern], dtype=bool)
    files = catalog.designs[held_out][int(rng.integers(len(catalog.designs[held_out])))]
    kicks, used = render_kicks(rng, bank, files, starts, in_roll, length)
    reference = rms(bank.get(files[0])[: int(BEAT * SAMPLE_RATE)])

    def at_time(sample: float) -> float:
        """The time in the finished drop of a sample position in the mix."""
        return round(sample / SAMPLE_RATE / tempo, 3)

    path = [s.path for s in bank.samples]
    info: dict[str, Any] = {
        "held_out": held_out,
        "bpm": round(BPM * tempo, 2),
        "kick_design": bank.samples[files[0]].design,
        "kick_files": [path[i] for i in used],
        "loops": [],
        "clap": None,
        "hits": [],
        "impact": None,
    }

    mix = kicks.copy()
    ducking = duck(length, starts, rng)
    layers = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.35, 0.35, 0.2]))
    for _ in range(layers if len(catalog.loops) else 0):
        index = int(rng.choice(catalog.loops))
        loop = render_loop(rng, bank.get(index), origin, length)
        cutoff = None
        if rng.random() < 0.85:  # the kick owns the sub band in a drop
            cutoff = rng.uniform(100.0, 250.0)
            loop = high_pass(loop, cutoff)
        level = rng.uniform(-18.0, 3.0 if cutoff else -6.0)
        gain = reference * db(level) / rms(loop)
        ducked = rng.random() < 0.85
        mix += gain * loop * (ducking if ducked else 1.0)
        info["loops"].append(
            {
                "path": path[index],
                "high_pass_hz": round(cutoff) if cutoff else None,
                "level_db": round(level, 1),  # relative to the kick
                "ducked": ducked,
            }
        )

    beat_samples = BEAT * SAMPLE_RATE
    if len(catalog.claps) and rng.random() < 0.6:
        index = int(rng.choice(catalog.claps))
        clap = bank.get(index)
        level = rng.uniform(-12.0, -2.0)
        gain = db(level) * reference / rms(clap[: len(clap) // 4 + 1])
        for beat in range(1, 4 * bars, 2):
            if rng.random() < 0.9:
                place(mix, clap, origin + round(beat * beat_samples), gain)
        info["clap"] = {"path": path[index], "level_db": round(level, 1)}
    hits = int(rng.poisson(8.0))
    for _ in range(hits if len(catalog.hits) else 0):
        index = int(rng.choice(catalog.hits))
        hit = bank.get(index)
        at = int(rng.integers(0, 4 * bars * 4)) * beat_samples / 4
        if rng.random() < 0.2:
            at += rng.uniform(-0.05, 0.05) * SAMPLE_RATE
        level = rng.uniform(-18.0, -3.0)
        gain = db(level) * reference / rms(hit[: len(hit) // 4 + 1])
        place(mix, hit, origin + round(at), gain)
        info["hits"].append(
            {
                "path": path[index],
                "time": at_time(origin + round(at)),
                "level_db": round(level, 1),
            }
        )
    info["hits"].sort(key=lambda hit: hit["time"])
    if len(catalog.impacts) and rng.random() < 0.3:
        index = int(rng.choice(catalog.impacts))
        impact = bank.get(index)
        level = rng.uniform(-12.0, 0.0)
        gain = db(level) * reference / rms(impact)
        at = origin + round(4 * beat_samples * rng.choice([0, 4]))
        place(mix, impact, at, gain)
        info["impact"] = {
            "path": path[index],
            "time": at_time(at),
            "level_db": round(level, 1),
        }

    mix, info["master"] = master(rng, mix)
    onsets = starts / SAMPLE_RATE
    if tempo != 1.0:
        n = int(len(mix) / tempo)
        mix = np.interp(np.arange(n) * tempo, np.arange(len(mix)), mix)
        onsets = onsets / tempo
    n = int(DROP_SECONDS * SAMPLE_RATE)
    mix = np.pad(mix[:n], (0, max(0, n - len(mix))))
    gain_db = rng.uniform(-6.0, 6.0)
    mix = normalize(mix) * db(gain_db)
    info["gain_db"] = round(gain_db, 1)
    onsets = onsets[(onsets >= 0.0) & (onsets < DROP_SECONDS)]
    info["kicks"] = len(onsets)
    return Drop(mix.astype(np.float32), onsets, info)
