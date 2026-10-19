import glob
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel

from src.core.config import CONFIG
from src.core.storage import load_audio, load_yaml
from src.models.audio import SampleKind, Signal
from src.services.synthesis.track import Track

AUDIO_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".aifc",
    ".au",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
}
HASH_BUCKETS = 1000
"""The held-out fraction is rounded to ``1 / HASH_BUCKETS``."""

CACHE_INDEX_FILE = "index.json"
CACHE_AUDIO_FILE = "audio.f32"


@dataclass(frozen=True)
class Sample:
    name: str
    path: Path
    signal: Signal
    kind: SampleKind

    bpm: int
    is_loop: bool

    @property
    def is_one_shot(self) -> bool:
        return not self.is_loop

    def stretch(self, track: Track) -> "Sample":
        rate = track.grid.bpm / self.bpm
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The `(hop_length|n_fft)` parameter is deprecated",
                category=FutureWarning,
            )
            stretched_signal = librosa.effects.time_stretch(
                self.signal, rate=rate
            )
        return replace(self, signal=stretched_signal, bpm=track.grid.bpm)


def is_held_out(name: str, fraction: float) -> bool:
    """A fixed choice of about ``fraction`` of the names, from a hash of the name."""
    return zlib.crc32(name.encode()) % HASH_BUCKETS < fraction * HASH_BUCKETS


@dataclass(frozen=True)
class SampleBank:
    samples: list[Sample]

    def select_kicks(self, held_out: bool, fraction: float) -> "SampleBank":
        """The bank with only the held-out kicks, or only the other kicks.

        The kicks are held out by ``is_held_out`` of their name. The samples of
        the other kinds do not change.
        """
        return SampleBank(
            samples=[
                sample
                for sample in self.samples
                if sample.kind != SampleKind.KICK
                or is_held_out(sample.name, fraction) == held_out
            ]
        )

    def draw(
        self,
        kind: SampleKind,
        rng: np.random.Generator,
        is_loop: bool | None = None,
    ) -> Sample:
        """Draw a sample of the kind at random (only loops or one-shots if set)."""
        pool = [
            sample
            for sample in self.samples
            if sample.kind == kind and is_loop in (None, sample.is_loop)
        ]
        if not pool:
            shape = {None: "", True: " loop", False: " one-shot"}[is_loop]
            raise ValueError(f"The sample bank has no {kind}{shape} samples.")
        return pool[rng.integers(len(pool))]


class SampleConfig(BaseModel):
    kind: SampleKind
    paths: list[Path]

    bpm: int = 160
    is_loop: bool = False


class SampleBankConfig(BaseModel):
    workspace: Path | None = None
    samples: list[SampleConfig]


class SampleFile(BaseModel):
    """An audio file that matches the bank config, before it is decoded."""

    kind: SampleKind
    path: Path
    name: str
    bpm: int
    is_loop: bool
    size: int
    modified_ns: int


class SampleBankCacheIndex(BaseModel):
    """The cached files. File ``i`` is from ``offsets[i]`` to ``offsets[i + 1]``."""

    sr: int
    files: list[SampleFile]
    offsets: list[int]


def find_sample_files(config: SampleBankConfig) -> list[SampleFile]:
    files: list[SampleFile] = []
    for sample_config in config.samples:
        for configured_path in sample_config.paths:
            path_pattern = (
                config.workspace / configured_path
                if config.workspace
                else configured_path
            )
            paths = sorted(
                Path(path)
                for path in glob.glob(str(path_pattern))
                if Path(path).suffix.lower() in AUDIO_EXTENSIONS
            )
            if not paths:
                raise FileNotFoundError(
                    f"No audio files matched sample path: {path_pattern}"
                )

            for path in paths:
                stat = path.stat()
                files.append(
                    SampleFile(
                        kind=sample_config.kind,
                        path=path,
                        name=_name_sample(path, config.workspace),
                        bpm=sample_config.bpm,
                        is_loop=sample_config.is_loop,
                        size=stat.st_size,
                        modified_ns=stat.st_mtime_ns,
                    )
                )
    return files


def _name_sample(path: Path, workspace: Path | None) -> str:
    """The path relative to the workspace, which is unique across packs."""
    return str(path.relative_to(workspace) if workspace else path)


def _create_sample(file: SampleFile, signal: Signal) -> Sample:
    return Sample(
        name=file.name,
        path=file.path,
        signal=signal,
        kind=file.kind,
        bpm=file.bpm,
        is_loop=file.is_loop,
    )


def _log_bank(sample_bank: SampleBank) -> None:
    counts = Counter(sample.kind for sample in sample_bank.samples)
    summary = ", ".join(f"{counts[kind]} {kind}" for kind in SampleKind)
    logger.info(f"Loaded sample bank: {summary}")


def load_sample_bank(config: SampleBankConfig) -> SampleBank:
    """Decode all of the samples into memory."""
    files = find_sample_files(config)
    logger.info(f"Decoding {len(files)} samples")
    sample_bank = SampleBank(
        samples=[_create_sample(file, load_audio(file.path)) for file in files]
    )
    _log_bank(sample_bank)
    return sample_bank


def _open_sample_bank_cache(
    folder: Path, files: list[SampleFile]
) -> SampleBank | None:
    """The cached bank, or None if the cache is missing or out of date."""
    try:
        index = SampleBankCacheIndex.model_validate_json(
            (folder / CACHE_INDEX_FILE).read_text()
        )
        audio = np.memmap(folder / CACHE_AUDIO_FILE, dtype=np.float32, mode="r")
    except (OSError, ValueError) as error:
        logger.debug(f"Could not open sample bank cache at {folder}: {error}")
        return None

    if index.sr != CONFIG.sr or index.files != files:
        return None
    samples = [
        _create_sample(file, audio[start:end])
        for file, start, end in zip(
            files, index.offsets[:-1], index.offsets[1:], strict=True
        )
    ]
    return SampleBank(samples=samples)


def _write_sample_bank_cache(folder: Path, files: list[SampleFile]) -> None:
    """Decode each file into the cache, one file at a time.

    The index is written last, so an interrupted write is not used.
    """
    if folder.is_file():
        folder.unlink()  # cache file of an older format
    folder.mkdir(parents=True, exist_ok=True)
    (folder / CACHE_INDEX_FILE).unlink(missing_ok=True)
    logger.info(f"Decoding {len(files)} samples into the cache at {folder}")
    offsets = [0]
    with open(folder / CACHE_AUDIO_FILE, "wb") as audio_file:
        for file in files:
            signal = np.asarray(load_audio(file.path), dtype=np.float32)
            audio_file.write(signal.tobytes())
            offsets.append(offsets[-1] + len(signal))
    index = SampleBankCacheIndex(sr=CONFIG.sr, files=files, offsets=offsets)
    (folder / CACHE_INDEX_FILE).write_text(index.model_dump_json())


def load_sample_bank_config(path: Path) -> SampleBankConfig:
    return SampleBankConfig.model_validate(load_yaml(path))


def load_sample_bank_from_file(
    path: Path, use_cache: bool = True
) -> SampleBank:
    """Load the bank of a config file.

    With ``use_cache``, the audio is read from a memory-mapped cache next to the
    config. The cache is rebuilt when the matched files or their settings change.
    """
    config = load_sample_bank_config(path)
    if not use_cache:
        return load_sample_bank(config)

    files = find_sample_files(config)
    cache_folder = path.with_suffix(f"{path.suffix}.cache")
    sample_bank = _open_sample_bank_cache(cache_folder, files)
    if sample_bank is None:
        _write_sample_bank_cache(cache_folder, files)
        sample_bank = _open_sample_bank_cache(cache_folder, files)
        assert sample_bank is not None
    _log_bank(sample_bank)
    return sample_bank
