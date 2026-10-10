import glob
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from src.core.storage import load_audio, load_yaml
from src.models.audio import Signal

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

SampleKind = Literal[
    "kick",
    "melody",
    "vocal",
    "drum",
    "bass",
    "impact",
    "fx",
]


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


@dataclass(frozen=True)
class SampleBank:
    samples: list[Sample]

    @property
    def kicks(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "kick"]

    @property
    def melodies(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "melody"]

    @property
    def vocals(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "vocal"]

    @property
    def drums(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "drum"]

    @property
    def basses(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "bass"]

    @property
    def impacts(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "impact"]

    @property
    def fxs(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "fx"]

    @property
    def loops(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.is_loop]

    @property
    def one_shots(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.is_one_shot]

    def draw_kick(self) -> Sample:
        """Draw a kick at random."""
        return random.choice(self.kicks)

    def draw_melodic_loop(self) -> Sample:
        """Draw a melodic loop at random."""
        return random.choice(
            [
                sample
                for sample in (self.melodies + self.vocals)
                if sample.is_loop
            ]
        )


class SampleConfig(BaseModel):
    kind: SampleKind
    paths: list[Path]

    bpm: int = 160
    is_loop: bool = False


class SampleBankConfig(BaseModel):
    workspace: Path | None = None
    samples: list[SampleConfig]


def load_sample_bank(config: SampleBankConfig) -> SampleBank:
    samples: list[Sample] = []
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
                name = path.stem
                signal = load_audio(path)
                sample = Sample(
                    name=name,
                    path=path,
                    signal=signal,
                    kind=sample_config.kind,
                    bpm=sample_config.bpm,
                    is_loop=sample_config.is_loop,
                )
                samples.append(sample)
    return SampleBank(samples=samples)


def _load_sample_bank_cache(
    path: Path, config_contents: bytes
) -> SampleBank | None:
    try:
        with open(path, "rb") as cache_file:
            cached_contents, sample_bank = pickle.load(cache_file)
    except Exception as error:
        logger.warning(f"Could not load sample bank cache from {path}: {error}")
        return None

    if cached_contents == config_contents and isinstance(
        sample_bank, SampleBank
    ):
        return sample_bank
    return None


def _save_sample_bank_cache(
    path: Path, config_contents: bytes, sample_bank: SampleBank
) -> None:
    try:
        with open(path, "wb") as cache_file:
            pickle.dump(
                (config_contents, sample_bank),
                cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception as error:
        logger.warning(f"Could not save sample bank cache to {path}: {error}")


def load_sample_bank_config(path: Path) -> SampleBankConfig:
    return SampleBankConfig.model_validate(load_yaml(path))


def load_sample_bank_from_file(
    path: Path, use_cache: bool = True
) -> SampleBank:
    config_contents = path.read_bytes()
    cache_path = path.with_suffix(f"{path.suffix}.cache")
    if use_cache and cache_path.exists():
        cached_bank = _load_sample_bank_cache(cache_path, config_contents)
        if cached_bank is not None:
            logger.info(f"Loaded sample bank from cache at {cache_path}")
            return cached_bank

    config = load_sample_bank_config(path)
    sample_bank = load_sample_bank(config)
    if use_cache:
        _save_sample_bank_cache(cache_path, config_contents, sample_bank)
    return sample_bank
