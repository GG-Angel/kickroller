from pathlib import Path

import numpy as np
from loguru import logger

from src.core.config import SynthConfig
from src.core.storage import save_audio, save_onsets
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.synth import generate_cropped_drop


def export_drops(
    bank: SampleBank,
    folder: Path,
    count: int,
    seed: int | None = None,
    *,
    config: SynthConfig,
    sr: int,
) -> None:
    """Write float32 WAV drops and onset CSVs. Draw and log a seed if unset."""
    seed_sequence = np.random.SeedSequence(seed)
    logger.info(f"Seed: {seed_sequence.entropy}")
    rng = np.random.default_rng(seed_sequence)
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        drop = generate_cropped_drop(bank, rng, config, sr)
        path = folder / f"drop_{index:03d}"
        save_audio(path.with_suffix(".wav"), drop.signal, sr)
        save_onsets(path.with_suffix(".csv"), drop.onsets)
