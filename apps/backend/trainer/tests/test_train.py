from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.config import CONFIG
from src.models.audio import SampleKind
from src.services.synthesis.bank import Sample, SampleBank
from src.services.training.network import load_checkpoint
from src.services.training.train import train


def _bank() -> SampleBank:
    rng = np.random.default_rng(0)

    def sample(name: str, kind: SampleKind) -> Sample:
        return Sample(
            name=name,
            path=Path(name),
            signal=rng.uniform(-0.5, 0.5, CONFIG.sr // 4).astype(np.float32),
            kind=kind,
            bpm=160,
            is_loop=False,
        )

    kicks = [sample(f"kick_{i}.wav", SampleKind.KICK) for i in range(20)]
    return SampleBank(samples=[*kicks, sample("clap.wav", SampleKind.DRUM)])


def test_train_saves_a_checkpoint(tmp_path: Path) -> None:
    config = CONFIG.train.model_copy(
        update={
            "steps": 2,
            "batch_size": 2,
            "validation_drops": 2,
            "stats_drops": 2,
            "validate_every": 1,
            "workers": 0,
        }
    )
    synth = CONFIG.synth.model_copy(
        update={"layers": [], "held_out_kick_fraction": 0.5}
    )
    output = tmp_path / "model.pt"

    with (
        patch(
            "src.services.training.train.load_sample_bank_from_file",
            return_value=_bank(),
        ),
        patch(
            "src.services.training.dataset.load_sample_bank_from_file",
            return_value=_bank(),
        ),
    ):
        best = train(
            Path("bank.yaml"), output, seed=0, config=config, synth=synth
        )

    assert 0 <= best.f_measure <= 1
    assert load_checkpoint(output).threshold == best.threshold
