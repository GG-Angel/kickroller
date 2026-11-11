from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.core.config import CONFIG
from src.models.audio import SampleKind
from src.services.scoring import evaluate
from src.services.synthesis.bank import Sample, SampleBank
from src.services.training.loop import TrainingBatch, _prepare_training_batch
from src.services.training.network import CONTEXT_FRAMES, load_checkpoint
from src.services.training.train import train


def _make_bank() -> SampleBank:
    rng = np.random.default_rng(0)

    def make_sample(name: str, kind: SampleKind) -> Sample:
        return Sample(
            name=name,
            path=Path(name),
            signal=rng.uniform(-0.5, 0.5, CONFIG.sr // 4).astype(np.float32),
            kind=kind,
            bpm=160,
            is_loop=False,
        )

    kicks = [make_sample(f"kick_{i}.wav", SampleKind.KICK) for i in range(20)]
    return SampleBank(
        samples=[*kicks, make_sample("clap.wav", SampleKind.DRUM)]
    )


@pytest.mark.parametrize("with_mask", [False, True])
def test_prepare_training_batch_preserves_inputs_and_masks_context(
    with_mask: bool,
) -> None:
    frames = 2 * CONTEXT_FRAMES + 10
    features = torch.ones(1, 3, 4, frames)
    targets = torch.ones(1, frames)
    supplied_mask = torch.zeros_like(targets)
    supplied_mask[:, CONTEXT_FRAMES : CONTEXT_FRAMES + 3] = 1
    original_mask = supplied_mask.clone()
    batch: TrainingBatch = (
        (features, targets, supplied_mask) if with_mask else (features, targets)
    )

    prepared_features, prepared_targets, mask = _prepare_training_batch(
        batch, torch.device("cpu")
    )

    torch.testing.assert_close(prepared_features, features)
    torch.testing.assert_close(prepared_targets, targets)
    expected_mask = (
        original_mask.clone() if with_mask else torch.ones_like(targets)
    )
    expected_mask[:, :CONTEXT_FRAMES] = 0
    expected_mask[:, -CONTEXT_FRAMES:] = 0
    torch.testing.assert_close(mask, expected_mask)
    torch.testing.assert_close(supplied_mask, original_mask)


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
            return_value=_make_bank(),
        ),
        patch(
            "src.services.training.dataset.load_sample_bank_from_file",
            return_value=_make_bank(),
        ),
        patch("src.services.training.train.evaluate", wraps=evaluate) as score,
    ):
        best = train(
            Path("bank.yaml"),
            output,
            seed=0,
            config=config,
            synth=synth,
            feature_config=CONFIG.features,
            sr=CONFIG.sr,
        )

    assert 0 <= best.f_measure <= 1
    assert score.call_args.kwargs["labeled_spans_only"] is False
    assert load_checkpoint(output).threshold == best.threshold
