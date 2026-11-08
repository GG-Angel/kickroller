from typing import Protocol

import numpy as np


class PatternGenerator(Protocol):
    def generate(self) -> np.ndarray: ...
