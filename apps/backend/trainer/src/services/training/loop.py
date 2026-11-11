from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter

import torch
from loguru import logger
from src.core.config import TrainConfig
from src.services.scoring import Evaluation
from src.services.training.network import (
    CONTEXT_FRAMES,
    KickNet,
    save_checkpoint,
)
from torch import Tensor

type TrainingBatch = tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]
type LossPolicy = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
type Validator = Callable[[], Evaluation]


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_training_batch(
    batch: TrainingBatch, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    if len(batch) == 2:
        features, targets = batch
        mask = None
    else:
        features, targets, mask = batch
    features = features.to(device)
    targets = targets.to(device)
    training_mask = (
        torch.ones_like(targets) if mask is None else mask.to(device).clone()
    )
    training_mask[:, :CONTEXT_FRAMES] = 0
    training_mask[:, -CONTEXT_FRAMES:] = 0
    return features, targets, training_mask


def run_training_loop(
    model: KickNet,
    batches: Iterable[TrainingBatch],
    output: Path,
    config: TrainConfig,
    loss_policy: LossPolicy,
    validate: Validator,
    initial_validation: Validator | None = None,
) -> Evaluation:
    """Train the model and save its best checkpoint."""
    device = _pick_device()
    model.to(device)
    positive_weight = torch.tensor(config.positive_weight, device=device)
    weights = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"Training {weights} weights on {device}.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )

    if initial_validation is None:
        best = Evaluation(-1.0, 0.0, 0.0, config.thresholds[0])
    else:
        best = initial_validation()
        save_checkpoint(output, model, best.threshold)
        logger.info(
            f"Baseline validation F {best.f_measure:.4f}, "
            f"threshold {best.threshold}, saved to {output}."
        )
    running, start = 0.0, perf_counter()
    model.train()
    for step, batch in enumerate(batches, 1):
        features, targets, mask = _prepare_training_batch(batch, device)
        logits = model(features)
        loss = loss_policy(logits, targets, mask, positive_weight)
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
            evaluation = validate()
            model.train()
            if evaluation.f_measure > best.f_measure:
                best = evaluation
                save_checkpoint(output, model, evaluation.threshold)
                saved = f", saved to {output}"
            else:
                saved = ""
            logger.info(
                f"Step {step}: validation F {evaluation.f_measure:.3f} "
                f"(precision {evaluation.precision:.3f}, "
                f"recall {evaluation.recall:.3f}, "
                f"threshold {evaluation.threshold}, "
                f"best F {best.f_measure:.3f}){saved}"
            )
        if step == config.steps:
            break
    return best
