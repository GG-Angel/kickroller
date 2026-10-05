from pathlib import Path

import librosa
import numpy as np
from loguru import logger

from core.settings import SETTINGS
from services.storage.io import save_audio_file

from .models import Bank, LabeledDrop
from .synth import create_drop


def _mix_clicks(drop: LabeledDrop) -> np.ndarray:
    """Return the drop audio mixed with clicks at each kick onset."""
    clicks = librosa.clicks(
        times=drop.onsets,
        sr=SETTINGS.sample_rate,
        length=len(drop.audio),
        click_freq=SETTINGS.synth.label_click_frequency_hz,
    )
    return 0.5 * drop.audio + 0.5 * clicks


def _write_drop(path: Path, drop: LabeledDrop, include_clicks: bool) -> None:
    """Write the drop to disk."""
    save_audio_file(path.with_suffix(".wav"), drop.audio)
    np.savetxt(path.with_suffix(".csv"), drop.onsets, fmt="%.4f")
    if include_clicks:
        save_audio_file(path.with_suffix(".clicks.wav"), _mix_clicks(drop))


def export_drops(
    bank: Bank,
    folder: Path,
    num_drops: int,
    seed: int | None = None,
    include_clicks: bool = False,
    held_out: bool = False,
) -> None:
    """Create and export synthetic drops (`held_out`: from held-out kick designs)."""
    seed_sequence = np.random.SeedSequence(seed)
    logger.info("Seed {seed}", seed=seed_sequence.entropy)
    rng = np.random.default_rng(seed_sequence)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(num_drops):
        drop = create_drop(bank, rng, held_out=held_out)
        name = f"drop_{i:03d}"
        _write_drop(folder / name, drop, include_clicks)
        logger.info(
            "Wrote {name}: {kicks} kicks at {bpm} BPM, kick design {design}",
            name=name,
            kicks=len(drop.onsets),
            bpm=drop.bpm,
            design=drop.kick_design,
        )
    logger.info("Wrote {count} drops to {folder}", count=num_drops, folder=folder)
