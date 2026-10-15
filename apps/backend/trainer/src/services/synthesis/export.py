from pathlib import Path

import numpy as np
from loguru import logger

from src.core.config import CONFIG
from src.core.storage import save_audio, save_onsets
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.crop import crop_drop, draw_crop
from src.services.synthesis.synth import generate_drop


def export_drops(
    bank: SampleBank, folder: Path, count: int, seed: int | None = None
) -> None:
    """Write ``count`` cropped drops and their kick onsets to ``folder``.

    Each drop is written as ``drop_000.wav`` with its onsets in ``drop_000.csv``.
    Without a seed, a new seed is drawn and logged, so the run can be repeated.
    """
    seed_sequence = np.random.SeedSequence(seed)
    logger.info(f"Seed {seed_sequence.entropy}")
    rng = np.random.default_rng(seed_sequence)
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        drop = generate_drop(bank, rng)
        duration = len(drop.signal) / CONFIG.sr
        window = draw_crop(rng, duration, CONFIG.synth.drop_seconds)
        drop = crop_drop(drop, window)
        path = folder / f"drop_{index:03d}"
        save_audio(path.with_suffix(".wav"), drop.signal)
        save_onsets(path.with_suffix(".csv"), drop.onsets)
