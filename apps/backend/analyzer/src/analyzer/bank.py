"""The On Point Samples packs, decoded once into a memory-mapped sample bank.

The bank has three kinds of samples:

- kick: complete kick one-shots. Each has a design name; the pitched versions of
  one kick have the same design, so that a held-out design is really unheard.
- loop: 160 BPM loops with no kicks (screeches, songstarter stems, atmospheres,
  top loops and fills). Each loop starts on a bar.
- hit: other one-shots (claps, snares, hats, percussion, FX, synth hits).
"""

import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from analyzer.audio import SAMPLE_RATE, decode_mid

KINDS = ("kick", "loop", "hit")

HE1 = "OPS - Hardstyle Essentials Vol. 1"
HE3 = "OPS - Hardstyle Essentials Vol. 3"
IR = "OPS - INDUSTRIAL RAWSTYLE PRODUCTION SUITE"
IRD = f"{IR}/OPS - Industrial Rawstyle Drum Expansion (Vol. 1)"
IRK = f"{IR}/OPS - Industrial Rawstyle Kick Expansion (Vol. 1)"
IRM = f"{IR}/OPS - Industrial Rawstyle Melody Vault"
IRP = f"{IR}/OPS - Industrial Rawstyle Predrops (Vol. 1)"
REV = "OPS - Rawphoric Essentials Vol. 1"
FR1 = "OPS - Hardstyle Freebie Vol. 1"
FR2 = "OPS - Hardstyle Freebie Vol. 2"

# (kind, glob relative to the packs folder). Globs ending in "/*" do not go
# into subfolders.
RULES: tuple[tuple[str, str], ...] = (
    ("kick", f"{HE3}/Kicks/**/*"),
    ("kick", f"{HE3}/Drums/Kicks/*"),
    ("kick", f"{IRK}/Kicks/*"),
    ("kick", f"{IRK}/Kicks (Kloenk)/*"),
    ("kick", f"{IRK}/Kicks (Various)/*"),
    ("kick", f"{IRD}/Kicks/*"),
    ("kick", f"{REV}/Drums/Kicks/*"),
    ("kick", f"{REV}/Kicks/Gated Kicks/*"),
    ("kick", f"{REV}/Kicks/Mid-Intro Kicks/*"),
    ("kick", f"{REV}/Kicks/Rawstyle Kicks/*"),
    ("kick", f"{REV}/Kicks/Rawphoric Kicks/*/*"),
    ("kick", f"{REV}/Kick Fundamentals/1. Kicks/**/*"),
    ("kick", f"{FR1}/Kicks/Kicks/*"),
    ("kick", f"{FR1}/Kicks/Pitched Raw Kick (A)/*"),
    ("kick", f"{FR1}/Kicks/Pitched Raw Kick (F#)/*"),
    ("kick", f"{FR1}/Kicks/Toks/*"),
    ("kick", f"{FR2}/Mid-Intro Kicks/*"),
    ("kick", f"{FR2}/Misc Kicks/*"),
    ("kick", f"{FR2}/Psy Kicks/*"),
    ("kick", f"{FR2}/Pitched Kicks/*/*"),
    ("kick", f"{FR2}/Kick Parts Folder/Creative Kicks/*"),
    ("kick", f"{FR2}/Kick Parts Folder/Lazer Kicks/*"),
    ("kick", f"{FR2}/Kick Parts Folder/Punchy Kicks/*"),
    ("kick", "OPS - Free SZP Type Psy Kicks/SZP Type Psy Kicks/*"),
    ("kick", "OPS - Free Zaag Kicks (Vol. 1)/Zaag Kicks/*"),
    ("loop", f"{IRM}/Screech Loops/*"),
    ("loop", f"{IRM}/Screech Loops/Dry/*"),
    ("loop", f"{IRM}/Songstarters/*/*"),
    ("loop", f"{IRM}/Songstarters/*/Dry/*"),
    ("loop", f"{IRM}/Atmospheres (Songstarter)/*"),
    ("loop", f"{IRM}/Atmospheres (Various)/*/*"),
    ("loop", f"{IRD}/_Top Loops/*"),
    ("loop", f"{IRD}/_Ride Loops/*"),
    ("loop", f"{IRP}/Snare Fills/*"),
    ("loop", f"{IRP}/Riser Fills/*"),
    ("hit", f"{HE3}/Drums/Claps/*"),
    ("hit", f"{HE3}/Drums/Hihats/*"),
    ("hit", f"{HE3}/Drums/Earcandy/*"),
    ("hit", f"{HE3}/FX/Distorted Snares/*"),
    ("hit", f"{HE3}/FX/Impacts/*"),
    ("hit", f"{HE3}/FX/Sub Drops/*"),
    ("hit", f"{HE3}/FX/Hardstyle Sounds/Distorted Snares/*"),
    ("hit", f"{HE3}/FX/Hardstyle Sounds/Distorted Sounds/*"),
    ("hit", f"{HE3}/FX/Hardstyle Sounds/Misc Sounds/*"),
    ("hit", f"{HE3}/Synths & Basses/Bass Hits/*"),
    ("hit", f"{HE3}/Synths & Basses/Stabs/*"),
    ("hit", f"{HE3}/Synths & Basses/Synth Stabs/*"),
    ("hit", f"{HE3}/Synths & Basses/War Horns/*"),
    ("hit", f"{HE1}/Renders; F-150bpm/*"),
    ("hit", f"{IRD}/Claps/*"),
    ("hit", f"{IRD}/Claps (Hard)/*"),
    ("hit", f"{IRD}/Hi Hats (Closed)/*"),
    ("hit", f"{IRD}/Open Hat/*"),
    ("hit", f"{IRD}/Percussion/*"),
    ("hit", f"{IRD}/Rides/*"),
    ("hit", f"{IRD}/Snares (*)/*"),
    ("hit", f"{REV}/Drums/Claps/*"),
    ("hit", f"{REV}/Drums/Hats/*"),
    ("hit", f"{REV}/Drums/Percussion/*"),
    ("hit", f"{REV}/Drums/Snares/*"),
    ("hit", f"{REV}/FX/Distorted Snares/*"),
    ("hit", f"{REV}/FX/Impacts/*"),
    ("hit", f"{FR1}/Drums/Claps/*"),
    ("hit", f"{FR1}/Drums/Hihats/*"),
    ("hit", f"{FR1}/Drums/Percussion/*"),
    ("hit", f"{FR1}/Drums/Snares/*"),
    ("hit", f"{FR1}/FX/Crashes/*"),
    ("hit", f"{FR1}/FX/Impacts/*"),
    ("hit", f"{FR1}/FX/Sub Drops/*"),
    ("hit", f"{FR1}/Synths & Melodic/Screeches/*"),
)

# File names that do not fit their folder: kick rolls and triplets (more than one
# kick), gated kicks (a rumble chopped into 1/16 pulses), songstarter mixes, stems with drums and bass stems (sub-bass notes can
# sound like kicks), and one-shots layered with a kick.
EXCLUDE = {
    "kick": re.compile(r"TRIPLET|ROLL|GATED", re.IGNORECASE),
    "loop": re.compile(r"_(FULL|DRUMS|KICK|BASS)\.WAV$", re.IGNORECASE),
    "hit": re.compile(r"KICK", re.IGNORECASE),
}

# Kick folders that hold the pitched versions of one design.
DESIGN_FOLDER = re.compile(r"^(.*Kick \d+.*|Pitched Raw Kick.*)$", re.IGNORECASE)
VARIATION_FOLDER = re.compile(r"^(High Variations|Original Kick|Main)$", re.IGNORECASE)
# Key and variation suffixes of pitched kick file names, for example "_D#_HIGH".
KEY_SUFFIX = re.compile(
    r"[ _]?([A-G](#|B)?\d?|\d+[A-G]#?)([ _](HIGH|LOW))?$", re.IGNORECASE
)

# The longest loop, kick and hit, in seconds.
MAX_LOOP_SECONDS = 24.0
MAX_ONESHOT_SECONDS = 2.0

INDEX, AUDIO = "index.json", "audio.npy"


@dataclass(frozen=True)
class Sample:
    kind: str
    design: str  # the same for all versions of one kick; the file for others
    path: str  # relative to the packs folder


@dataclass
class Bank:
    """Decoded mono samples at `SAMPLE_RATE`, float16, one after the other."""

    audio: np.ndarray
    offsets: np.ndarray  # start of each sample in `audio`, and the end
    samples: list[Sample]

    def get(self, index: int) -> np.ndarray:
        return self.audio[self.offsets[index] : self.offsets[index + 1]].astype(
            np.float32
        )

    def indices(self, kind: str) -> np.ndarray:
        return np.array([i for i, s in enumerate(self.samples) if s.kind == kind])


def design_name(path: Path) -> str:
    folder = path.parent
    if VARIATION_FOLDER.match(folder.name):
        folder = folder.parent
    if DESIGN_FOLDER.match(folder.name):
        return folder.as_posix()
    stem = path.stem.lstrip("_")
    return f"{folder.as_posix()}/{KEY_SUFFIX.sub('', stem).upper()}"


def catalog(root: Path) -> list[Sample]:
    """All samples of the rules that exist under `root`, sorted by path."""
    found: dict[str, Sample] = {}
    for kind, pattern in RULES:
        for path in root.glob(pattern, case_sensitive=False):
            relative = path.relative_to(root)
            if path.suffix.lower() != ".wav" or EXCLUDE[kind].search(path.name):
                continue
            design = design_name(relative) if kind == "kick" else relative.as_posix()
            found.setdefault(
                relative.as_posix(), Sample(kind, design, relative.as_posix())
            )
    return sorted(found.values(), key=lambda s: s.path)


def is_held_out(design: str, fraction: float) -> bool:
    """A fixed choice of about `fraction` of the designs, from the design name."""
    return zlib.crc32(design.encode()) % 1000 < fraction * 1000


def trim(signal: np.ndarray, kind: str, sample_rate: int) -> np.ndarray:
    """Remove leading silence of one-shots (so time zero is the attack) and cut to length."""
    if kind == "loop":
        return signal[: int(MAX_LOOP_SECONDS * sample_rate)]
    peak = float(np.max(np.abs(signal), initial=0.0))
    loud = np.flatnonzero(np.abs(signal) >= 0.01 * peak)
    start = int(loud[0]) if len(loud) else 0
    return signal[start : start + int(MAX_ONESHOT_SECONDS * sample_rate)]


def load_bank(cache: Path) -> Bank:
    """The bank in `cache`, with the audio memory-mapped (not read into memory)."""
    index = json.loads((cache / INDEX).read_text())
    return Bank(
        np.load(cache / AUDIO, mmap_mode="r"),
        np.array(index["offsets"]),
        [Sample(**s) for s in index["samples"]],
    )


def build_bank(root: Path, cache: Path, workers: int = 8) -> Bank:
    """Decode all samples to `cache` (a folder), or load the bank from it."""
    samples = catalog(root)
    if (cache / INDEX).exists() and (cache / AUDIO).exists():
        bank = load_bank(cache)
        if bank.samples == samples:
            logger.info(
                "Loaded the sample bank from {cache} ({count} samples)",
                cache=cache,
                count=len(samples),
            )
            return bank
    if not samples:
        raise RuntimeError(f"no On Point samples found under {root}")

    counts = {kind: sum(s.kind == kind for s in samples) for kind in KINDS}
    logger.info(
        "Decoding {total} samples ({kicks} kicks in {designs} designs, {loops} loops, "
        "{hits} hits)",
        total=len(samples),
        kicks=counts["kick"],
        designs=len({s.design for s in samples if s.kind == "kick"}),
        loops=counts["loop"],
        hits=counts["hit"],
    )

    def decode(sample: Sample) -> np.ndarray:
        signal = decode_mid(root / sample.path, SAMPLE_RATE)
        return trim(signal, sample.kind, SAMPLE_RATE).astype(np.float16)

    with ThreadPoolExecutor(workers) as pool:
        decoded = list(pool.map(decode, samples))
    offsets = np.concatenate([[0], np.cumsum([len(d) for d in decoded])])
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / AUDIO, np.concatenate(decoded))
    (cache / INDEX).write_text(
        json.dumps(
            {"samples": [s.__dict__ for s in samples], "offsets": offsets.tolist()}
        )
    )
    logger.info(
        "Wrote the sample bank to {cache} ({seconds:.0f} s of audio)",
        cache=cache,
        seconds=offsets[-1] / SAMPLE_RATE,
    )
    return load_bank(cache)
