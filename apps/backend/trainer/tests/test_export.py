from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from src.core.config import CONFIG
from src.models.audio import LabeledDrop
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.export import export_drops


def _drop() -> LabeledDrop:
    return LabeledDrop(
        signal=np.zeros(13 * CONFIG.sr, dtype=np.float32),
        onsets=np.arange(0.0, 13.0, 0.375),
    )


def test_export_drops_writes_numbered_audio_and_onsets(tmp_path: Path) -> None:
    folder = tmp_path / "drops"

    with patch(
        "src.services.synthesis.export.generate_drop", return_value=_drop()
    ):
        export_drops(SampleBank(samples=[]), folder, count=2, seed=0)

    assert sorted(path.name for path in folder.iterdir()) == [
        "drop_000.csv",
        "drop_000.wav",
        "drop_001.csv",
        "drop_001.wav",
    ]
    onsets = np.loadtxt(folder / "drop_000.csv")
    assert len(onsets) > 0
    assert 0 <= onsets.min() and onsets.max() < CONFIG.synth.drop_seconds


def test_export_drops_normalizes_the_peak(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    drop = LabeledDrop(
        signal=rng.uniform(-3, 3, 13 * CONFIG.sr).astype(np.float32),
        onsets=np.arange(0.0, 13.0, 0.375),
    )

    with patch(
        "src.services.synthesis.export.generate_drop", return_value=drop
    ):
        export_drops(SampleBank(samples=[]), tmp_path, count=1, seed=0)

    _, signal = wavfile.read(tmp_path / "drop_000.wav")
    peak = 10 ** (CONFIG.synth.peak_db / 20)
    assert np.abs(signal).max() == pytest.approx(peak, rel=1e-6)
