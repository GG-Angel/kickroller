from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.core.config import CONFIG, SynthConfig, TrainConfig
from src.services.synthesis.bank import SampleBank, load_sample_bank_from_file
from src.services.synthesis.synth import generate_cropped_drop
from src.services.training.dataset import DropStream, Example, make_example
from src.services.training.evaluate import Evaluation, evaluate, predict
from src.services.training.network import (
    CONTEXT_FRAMES,
    KickNet,
    save_checkpoint,
)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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

    device = _pick_device()
    model = KickNet()
    stats = _make_examples(
        bank.select_kicks(False, fraction),
        config.stats_drops,
        np.random.default_rng(stats_seed),
        synth,
    )
    model.set_standardization(torch.stack([s.features for s in stats]))
    model.to(device)
    weights = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Training {weights} weights on {device}.")

    loader = DataLoader(
        DropStream(bank_file, use_cache, stream_seed, synth),
        batch_size=config.batch_size,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
    positive_weight = torch.tensor(config.positive_weight, device=device)
    best = Evaluation(-1.0, 0.0, 0.0, config.thresholds[0])
    running, start = 0.0, perf_counter()
    model.train()
    for step, (features, targets) in enumerate(loader, 1):
        logits = model(features.to(device))
        # the edge frames do not see their full context
        mask = torch.ones_like(logits)
        mask[:, :CONTEXT_FRAMES] = 0
        mask[:, -CONTEXT_FRAMES:] = 0
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.to(device), weight=mask, pos_weight=positive_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        if step % config.log_every == 0:
            logger.info(
                f"Step {step}/{config.steps}: loss "
                f"{running / config.log_every:.4f}, "
                f"{step / (perf_counter() - start):.2f} steps/s"
            )
            running = 0.0
        if step % config.validate_every == 0 or step == config.steps:
            probabilities = predict(
                model, validation_features, config.batch_size
            )
            evaluation = evaluate(probabilities, validation_onsets, config)
            model.train()
            improved = evaluation.f_measure > best.f_measure
            if improved:
                best = evaluation
                save_checkpoint(output, model, evaluation.threshold)
            logger.info(
                f"Step {step}: validation F {evaluation.f_measure:.3f} "
                f"(precision {evaluation.precision:.3f}, "
                f"recall {evaluation.recall:.3f}, "
                f"threshold {evaluation.threshold})"
                + (f", saved to {output}" if improved else "")
            )
        if step == config.steps:
            break
    return best
