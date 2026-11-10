from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.core.config import CONFIG, SynthConfig, TrainConfig
from src.services.scoring import Evaluation, evaluate, predict
from src.services.synthesis.bank import SampleBank, load_sample_bank_from_file
from src.services.synthesis.synth import generate_cropped_drop
from src.services.training.dataset import DropStream, Example, make_example
from src.services.training.loop import train_loop
from src.services.training.network import KickNet


def _synthetic_loss_policy(
    logits: Tensor, targets: Tensor, mask: Tensor, positive_weight: Tensor
) -> Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, targets, weight=mask, pos_weight=positive_weight
    )


def _make_examples(
    bank: SampleBank, count: int, rng: np.random.Generator, config: SynthConfig
) -> list[Example]:
    return [
        make_example(generate_cropped_drop(bank, rng, config))
        for _ in range(count)
    ]


def train(
    bank_file: Path,
    output: Path,
    use_cache: bool = True,
    seed: int | None = None,
    config: TrainConfig = CONFIG.train,
    synth: SynthConfig = CONFIG.synth,
) -> Evaluation:
    """Train a KickNet and save the version with the best validation F-measure.

    Validation uses a fixed set of drops with the held-out kicks only.
    Without a seed, a new seed is drawn and logged.
    """
    seed_sequence = np.random.SeedSequence(seed)
    logger.info(f"Seed: {seed_sequence.entropy}")
    torch_seed, stats_seed, stream_seed = map(
        int, seed_sequence.generate_state(3)
    )
    torch.manual_seed(torch_seed)
    bank = load_sample_bank_from_file(bank_file, use_cache=use_cache)
    fraction = synth.held_out_kick_fraction

    rng = np.random.default_rng(config.validation_seed)
    held_out = bank.select_kicks(True, fraction)
    validation_drops = [
        generate_cropped_drop(held_out, rng, synth)
        for _ in range(config.validation_drops)
    ]
    validation_features = torch.stack(
        [make_example(drop).features for drop in validation_drops]
    )
    validation_onsets = [drop.onsets for drop in validation_drops]
    logger.info(
        f"Made {len(validation_drops)} validation drops with "
        f"{sum(len(onsets) for onsets in validation_onsets)} held-out kicks."
    )

    model = KickNet()
    stats = _make_examples(
        bank.select_kicks(False, fraction),
        config.stats_drops,
        np.random.default_rng(stats_seed),
        synth,
    )
    model.set_standardization(
        torch.stack([example.features for example in stats])
    )

    loader = DataLoader(
        DropStream(bank_file, use_cache, stream_seed, synth),
        batch_size=config.batch_size,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )

    def validate() -> Evaluation:
        probabilities = predict(model, validation_features, config.batch_size)
        return evaluate(probabilities, validation_onsets, config)

    return train_loop(
        model,
        loader,
        output,
        config,
        _synthetic_loss_policy,
        validate,
    )
