"""The sample bank: the files of a bank config, decoded once into a cache folder.

The cache has the audio of all files one after the other (float16, read as a
memory map, so it is not loaded into memory) and an index of the files. If the
files of the config or the sample rate change, the cache is made again.
"""

import re
import tomllib
from itertools import pairwise
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import ValidationError

from core.settings import SETTINGS
from services.storage.io import load_audio_file

from .models import Bank, BankCacheIndex, BankConfig, Sample, SampleFile

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}
CACHE_INDEX_FILE = "index.json"
CACHE_AUDIO_FILE = "audio.f16"
CACHE_DTYPE = np.dtype(np.float16)


def find_sample_files(config: BankConfig) -> list[SampleFile]:
    """The audio files of each sample group in `config`, in a fixed order."""
    files: list[SampleFile] = []
    for group in config.samples:
        exclude = re.compile(group.exclude, re.IGNORECASE) if group.exclude else None
        for pattern in group.files:
            for path in sorted(config.root.glob(pattern.as_posix())):
                if (
                    not path.is_file()
                    or path.suffix.lower() not in AUDIO_SUFFIXES
                    or (exclude and exclude.search(path.name))
                ):
                    continue  # skip non-audio or excluded files
                files.append(SampleFile(kind=group.kind, path=path, bpm=group.bpm))
    return files


def _read_cache_index(cache: Path) -> BankCacheIndex | None:
    try:
        return BankCacheIndex.model_validate_json(
            (cache / CACHE_INDEX_FILE).read_text()
        )
    except (OSError, ValidationError):
        return None


def _is_cache_current(cache: Path, files: list[SampleFile]) -> bool:
    """Whether the cache has all of `files`, at the current sample rate."""
    index = _read_cache_index(cache)
    audio = cache / CACHE_AUDIO_FILE
    return (
        index is not None
        and index.sample_rate == SETTINGS.sample_rate
        and index.files == files
        and audio.is_file()
        and audio.stat().st_size == index.offsets[-1] * CACHE_DTYPE.itemsize
    )


def _write_cache(cache: Path, files: list[SampleFile]) -> None:
    """Decode `files` one at a time into the cache, then write the index."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / CACHE_INDEX_FILE).unlink(missing_ok=True)  # invalid until it is written
    offsets = [0]
    with (cache / CACHE_AUDIO_FILE).open("wb") as out:
        for file in files:
            audio = load_audio_file(file.path).astype(CACHE_DTYPE)
            audio.tofile(out)
            offsets.append(offsets[-1] + len(audio))
    index = BankCacheIndex(
        sample_rate=SETTINGS.sample_rate, files=files, offsets=offsets
    )
    (cache / CACHE_INDEX_FILE).write_text(index.model_dump_json())


def _open_cache(cache: Path) -> list[Sample]:
    """The samples in the cache, with their audio memory-mapped."""
    index = BankCacheIndex.model_validate_json((cache / CACHE_INDEX_FILE).read_text())
    audio = np.memmap(cache / CACHE_AUDIO_FILE, dtype=CACHE_DTYPE, mode="r")
    return [
        Sample(
            kind=file.kind, path=file.path, bpm=file.bpm, cached_audio=audio[start:end]
        )
        for file, (start, end) in zip(index.files, pairwise(index.offsets), strict=True)
    ]


def load_bank(config: BankConfig, cache: Path) -> Bank:
    """The samples of `config`. They are decoded into `cache` only if it is not current."""
    files = find_sample_files(config)
    if not files:
        raise ValueError(f"No sample files found in {config.root}")
    if _is_cache_current(cache, files):
        logger.info("Loading the bank from {cache}", cache=cache)
    else:
        logger.info(
            "Decoding {count} samples into {cache}", count=len(files), cache=cache
        )
        _write_cache(cache, files)
    samples = _open_cache(cache)
    return Bank(
        kicks=[s for s in samples if s.kind == "kick"],
        loops=[s for s in samples if s.kind == "loop"],
        claps=[s for s in samples if s.kind == "clap"],
        impacts=[s for s in samples if s.kind == "impact"],
        hits=[s for s in samples if s.kind == "hit"],
    )


def load_bank_from_file(path: Path, cache: Path) -> Bank:
    config = BankConfig.model_validate(tomllib.loads(path.read_text()))
    bank = load_bank(config, cache)
    logger.info(
        "Loaded bank: {kicks} kicks, {loops} loops, {claps} claps, {hits} hits, {impacts} impacts",
        kicks=len(bank.kicks),
        loops=len(bank.loops),
        claps=len(bank.claps),
        hits=len(bank.hits),
        impacts=len(bank.impacts),
    )
    return bank
