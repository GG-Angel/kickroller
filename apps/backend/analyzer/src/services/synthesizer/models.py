from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

SampleKind = Literal["kick", "loop", "clap", "impact", "hit"]


class SampleConfig(BaseModel):
    kind: SampleKind
    files: list[Path] = Field(default_factory=list)
    bpm: float = 160.0
    exclude: str | None = None


class BankConfig(BaseModel):
    root: Path = Path(".")
    samples: list[SampleConfig] = Field(default_factory=list)


@dataclass(frozen=True)
class Sample:
    kind: SampleKind
    path: Path
    bpm: float
    audio: np.ndarray


@dataclass(frozen=True)
class Bank:
    kicks: list[Sample]
    loops: list[Sample]
    claps: list[Sample]
    impacts: list[Sample]
    hits: list[Sample]


@dataclass(frozen=True)
class LabeledDrop:
    audio: np.ndarray
    onsets: np.ndarray
