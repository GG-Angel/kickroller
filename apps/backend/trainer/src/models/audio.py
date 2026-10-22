from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float32]


class SampleKind(StrEnum):
    KICK = "kick"
    MELODY = "melody"
    VOCAL = "vocal"
    DRUM = "drum"
    IMPACT = "impact"
    FX = "fx"
    BASS = "bass"


@dataclass(frozen=True)
class LabeledDrop:
    signal: Signal
    onsets: NDArray[np.float64]
    """Kick onsets in seconds."""
