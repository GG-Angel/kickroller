from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

SampleKind = Literal["kick", "loop", "clap", "impact", "hit"]


@dataclass(frozen=True)
class Sample:
    kind: SampleKind
    path: Path
    bpm: float
    audio: np.ndarray


class BankSampleSettings(BaseModel):
    kind: SampleKind
    files: list[Path] = Field(default_factory=list)
    exclude: str = ""
    bpm: float = 160.0


class BankSettings(BaseModel):
    root: Path
    samples: list[BankSampleSettings] = Field(default_factory=list)


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
