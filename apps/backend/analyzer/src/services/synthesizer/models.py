from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

SampleKind = Literal["kick", "loop", "clap", "impact", "hit"]


class SampleConfig(BaseModel):
    kind: SampleKind
    files: list[Path] = Field(default_factory=list)
    bpm: float = 160.0
    exclude: str | None = None


class BankConfig(BaseModel):
    root: Path = Path(".")
    kick_design_folders: list[Path] = Field(default_factory=list)
    samples: list[SampleConfig] = Field(default_factory=list)


class SampleFile(BaseModel):
    """An audio file of the bank, before it is decoded."""

    model_config = ConfigDict(frozen=True)

    kind: SampleKind
    path: Path
    bpm: float


class BankCacheIndex(BaseModel):
    """The files in the bank cache. The audio of file i is from offsets[i] to offsets[i + 1]."""

    sample_rate: int
    files: list[SampleFile]
    offsets: list[int]


@dataclass(frozen=True)
class Sample:
    kind: SampleKind
    path: Path
    name: str  # the path relative to the bank root
    bpm: float
    cached_audio: np.ndarray  # float16, memory-mapped from the bank cache

    @property
    def audio(self) -> np.ndarray:
        """The audio as a float32 copy."""
        return self.cached_audio.astype(np.float32)


@dataclass(frozen=True)
class KickDesign:
    """All versions (keys) of one kick. The name is a path relative to the bank root."""

    name: str
    kicks: list[Sample]


@dataclass(frozen=True)
class Bank:
    kick_designs: list[KickDesign]
    loops: list[Sample]
    claps: list[Sample]
    impacts: list[Sample]
    hits: list[Sample]


@dataclass(frozen=True)
class LabeledDrop:
    audio: np.ndarray
    onsets: np.ndarray
    bpm: int
    kick_design: str
