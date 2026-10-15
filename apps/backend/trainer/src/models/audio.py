from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float32]


@dataclass(frozen=True)
class LabeledDrop:
    signal: Signal
    onsets: NDArray[np.float64]
    """Kick onsets in seconds."""
