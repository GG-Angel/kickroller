import numpy as np

from src.core.config import MasterConfig
from src.services.synthesis.effects import (
    Clip,
    Effect,
    Gain,
    Limit,
    Mp3,
    Normalize,
    Tilt,
)


def draw_master_effects(
    config: MasterConfig | None, peak_db: float, rng: np.random.Generator
) -> list[Effect]:
    """The master effects of one drop.

    The first effect sets the peak to ``peak_db``, so that the drives of the
    clip and the limiter do not depend on the level of the mix. The random
    effects of ``config`` follow, if it is not None.
    """
    effects: list[Effect] = [Normalize(peak_db)]
    if config is None:
        return effects
    if rng.random() < config.tilt_probability:
        db = rng.uniform(*config.tilt_db_range)
        effects.append(Tilt(db, config.tilt_pivot_hz))
    if rng.random() < config.clip_probability:
        effects.append(Clip(rng.uniform(*config.clip_drive_db_range)))
    if rng.random() < config.limit_probability:
        drive_db = rng.uniform(*config.limit_drive_db_range)
        effects.append(Limit(drive_db, config.limit_ceiling_db))
    if rng.random() < config.mp3_probability:
        effects.append(Mp3(rng.uniform(*config.mp3_compression_range)))
    effects.append(Gain(rng.uniform(*config.gain_db_range)))
    return effects
