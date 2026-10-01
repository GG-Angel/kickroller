"""A sample bank: the samples of a bank config, decoded once and memory-mapped.

The config is a TOML file (see bank.example.toml). It gives a root folder and,
for each kind of sample, globs relative to the root:

- kick: complete kick one-shots. Each has a design; the pitched versions of
  one kick have the same design, so that a held-out design is really unheard.
- loop: loops with no kicks and no sub-bass (screeches, atmospheres, top loops,
  fills). Each loop starts on a bar.
- clap: claps, put on beats 2 and 4.
- impact: impacts, crashes and sub drops, put at the start of a bar.
- hit: other one-shots (snares, hats, percussion, FX, synth hits).

Samples are at BPM, or at the tempo that `bpm` gives for their glob (in any
kind). These are time-stretched to BPM, with the same pitch, when decoded.
"""

import json
import re
import tomllib
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from analyzer.audio.io import SAMPLE_RATE, decode_mid

KINDS = ("kick", "loop", "clap", "impact", "hit")
AUDIO_SUFFIXES = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a")
BPM = 160.0  # the tempo of the samples in the bank (synth.py resamples the loops)

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
    path: str  # relative to the bank root
    bpm: float | None = None  # the tempo of a file that is not at BPM


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
        return np.array(
            [i for i, s in enumerate(self.samples) if s.kind == kind], dtype=int
        )


@dataclass(frozen=True)
class Source:
    """The samples of one kind in a bank config."""

    files: tuple[str, ...] = ()  # globs of audio files
    exclude: re.Pattern[str] | None = None  # file names to skip
    tempos: tuple[tuple[str, float], ...] = ()  # (glob, tempo); first match counts


@dataclass(frozen=True)
class BankConfig:
    root: Path
    sources: dict[str, Source]
    designs: tuple[str, ...] = ()  # globs of folders that each hold one kick design


def globs(value: object, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise ValueError(f"{where} must be a glob or a list of globs")


def tempos(value: object, kind: str) -> tuple[tuple[str, float], ...]:
    """The `bpm` table of a kind. Tempos are BPM/2 to 2*BPM, a stretch that keeps quality."""
    if isinstance(value, dict) and all(
        isinstance(v, int | float)
        and not isinstance(v, bool)
        and BPM / 2 <= v <= 2 * BPM
        for v in value.values()
    ):
        return tuple((str(glob), float(bpm)) for glob, bpm in value.items())
    raise ValueError(
        f"[{kind}] bpm must be a table of globs and tempos from {BPM / 2:.0f} "
        f'to {2 * BPM:.0f}, for example {{ "loops/150/*" = 150 }}'
    )


def read_config(path: Path) -> BankConfig:
    """The bank config in the TOML file `path`. A relative root is relative to the file."""
    data = tomllib.loads(path.read_text())
    unknown = sorted(set(data) - {"root", *KINDS})
    if unknown:
        raise ValueError(f"unknown keys {unknown} (use root, {', '.join(KINDS)})")
    sources = {}
    for kind in KINDS:
        table = data.get(kind, {})
        keys = ("files", "exclude", "bpm") + (("designs",) if kind == "kick" else ())
        unknown = sorted(set(table) - set(keys))
        if unknown:
            raise ValueError(
                f"unknown keys {unknown} in [{kind}] (use {', '.join(keys)})"
            )
        try:
            exclude = re.compile(table["exclude"], re.IGNORECASE)
        except KeyError:
            exclude = None
        except (re.error, TypeError) as error:
            raise ValueError(f"[{kind}] exclude: {error}") from error
        sources[kind] = Source(
            globs(table.get("files", []), f"[{kind}] files"),
            exclude,
            tempos(table["bpm"], kind) if "bpm" in table else (),
        )
    if not sources["kick"].files:
        raise ValueError("[kick] has no files")
    root = path.parent / Path(data.get("root", ".")).expanduser()
    designs = globs(data.get("kick", {}).get("designs", []), "[kick] designs")
    return BankConfig(root, sources, designs)


def design_name(path: Path) -> str:
    """The folder and the file name without its key suffix ("KICK 3_F#" is "KICK 3")."""
    stem = KEY_SUFFIX.sub("", path.stem.lstrip("_")).upper()
    return f"{path.parent.as_posix()}/{stem}"


def catalog(config: BankConfig) -> list[Sample]:
    """All samples of the config, sorted by path.

    A file found for more than one kind gets the first kind in KINDS. The design
    of a kick is its nearest design folder, or else the design of its name. A
    sample gets the tempo of the first matching `bpm` glob of its kind.
    """
    root = config.root
    folders = {
        folder.relative_to(root)
        for pattern in config.designs
        for folder in root.glob(pattern, case_sensitive=False)
        if folder.is_dir()
    }

    def design(path: Path) -> str:
        folder = next((p for p in path.parents if p in folders), None)
        return folder.as_posix() if folder else design_name(path)

    matched: set[tuple[str, str]] = set()

    def tempo(path: Path, kind: str) -> float | None:
        """The tempo of a file from the first matching `bpm` glob, if not BPM."""
        for pattern, bpm in config.sources[kind].tempos:
            if path.full_match(pattern, case_sensitive=False):
                matched.add((kind, pattern))
                return None if bpm == BPM else bpm
        return None

    found: dict[str, Sample] = {}
    for kind, source in config.sources.items():
        for pattern in source.files:
            for path in root.glob(pattern, case_sensitive=False):
                if (
                    path.suffix.lower() not in AUDIO_SUFFIXES
                    or not path.is_file()
                    or (source.exclude and source.exclude.search(path.name))
                ):
                    continue
                relative = path.relative_to(root)
                name = relative.as_posix()
                if name in found:
                    continue
                found[name] = Sample(
                    kind,
                    design(relative) if kind == "kick" else name,
                    name,
                    tempo(relative, kind),
                )
    for kind, source in config.sources.items():
        for pattern, _ in source.tempos:
            if (kind, pattern) not in matched:
                logger.warning(
                    "[{kind}] bpm: {pattern} matches no {kind} file",
                    kind=kind,
                    pattern=pattern,
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


def build_bank(config: BankConfig, cache: Path, workers: int = 8) -> Bank:
    """Decode all samples of `config` to `cache` (a folder), or load the bank from it."""
    root = config.root
    if not root.is_dir():
        raise FileNotFoundError(f"the bank root {root} is not a folder")
    samples = catalog(config)
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
        raise RuntimeError(f"no samples of the bank config found under {root}")

    counts = {kind: sum(s.kind == kind for s in samples) for kind in KINDS}
    logger.info(
        "Decoding {total} samples ({kicks} kicks in {designs} designs, {loops} loops, "
        "{claps} claps, {impacts} impacts, {hits} other hits)",
        total=len(samples),
        kicks=counts["kick"],
        designs=len({s.design for s in samples if s.kind == "kick"}),
        loops=counts["loop"],
        claps=counts["clap"],
        impacts=counts["impact"],
        hits=counts["hit"],
    )

    stretched = sum(s.bpm is not None for s in samples)
    if stretched:
        logger.info(
            "Time-stretching {count} samples to {bpm:.0f} BPM (same pitch)",
            count=stretched,
            bpm=BPM,
        )

    def decode(sample: Sample) -> np.ndarray:
        tempo = BPM / sample.bpm if sample.bpm is not None else 1.0
        signal = decode_mid(root / sample.path, SAMPLE_RATE, tempo)
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
