from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mir_eval
import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.core.config import CONFIG, SynthConfig, TrainConfig
from src.core.storage import load_audio, load_onsets
from src.services.analysis.analyze import select_labeled_onsets
from src.services.training.dataset import DropStream
from src.services.training.evaluate import Evaluation, pick_onsets, predict
from src.services.training.features import compute_features, make_targets
from src.services.training.network import (
    CONTEXT_FRAMES,
    load_checkpoint,
    save_checkpoint,
)
from src.services.training.train import _pick_device


@dataclass(frozen=True)
class RealTrack:
    features: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    onsets: np.ndarray


def load_real_track(path: Path) -> RealTrack:
    """Load a track, masking sections with no labels out of the loss."""
    candidates = [
        path.with_suffix(".csv"),
        path.with_name(f"{path.stem}-labels.csv"),
    ]
    labels = [candidate for candidate in candidates if candidate.exists()]
    if len(labels) != 1:
        raise ValueError(f"Expected one label CSV for {path}, got {labels}.")
    onsets = load_onsets(labels[0])
    signal = load_audio(path)
    if (
        not len(onsets)
        or not np.isfinite(onsets).all()
        or (onsets < 0).any()
        or (np.diff(onsets) <= 0).any()
        or onsets[-1] > len(signal) / CONFIG.sr
    ):
        raise ValueError(f"Invalid or empty onset labels in {labels[0]}.")
    features = compute_features(signal)
    times = np.arange(features.shape[-1]) * CONFIG.features.hop / CONFIG.sr
    selected = select_labeled_onsets(times, onsets)
    mask = np.isin(times, selected).astype(np.float32)
    targets = make_targets(onsets, len(times))
    return RealTrack(
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(mask),
        onsets,
    )


def _real_crop(
    track: RealTrack, frames: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop around a scored frame, keeping its full network context."""
    if frames <= 2 * CONTEXT_FRAMES:
        raise ValueError("A crop must be longer than the network context.")
    if track.features.shape[-1] < frames:
        raise ValueError("A real track is shorter than the training crop.")
    candidates = torch.nonzero(track.mask).flatten().numpy()
    if not len(candidates):
        raise ValueError("A real track has no labeled frames.")
    anchor = int(rng.choice(candidates))
    offset = int(rng.integers(CONTEXT_FRAMES, frames - CONTEXT_FRAMES))
    start = int(np.clip(anchor - offset, 0, track.features.shape[-1] - frames))
    end = start + frames
    return (
        track.features[..., start:end],
        track.targets[start:end],
        track.mask[start:end],
    )


def evaluate_real(
    probabilities: list[np.ndarray],
    tracks: list[RealTrack],
    config: TrainConfig = CONFIG.train,
) -> Evaluation:
    """Select a threshold using pooled counts inside labeled spans only."""
    if not tracks:
        raise ValueError("Real validation needs at least one track.")
    references = sum(len(track.onsets) for track in tracks)
    best = Evaluation(-1.0, 0.0, 0.0, config.thresholds[0])
    for threshold in config.thresholds:
        detected = hits = 0
        for probability, track in zip(probabilities, tracks, strict=True):
            estimated = select_labeled_onsets(
                pick_onsets(probability, threshold, config),
                track.onsets,
                config.tolerance,
            )
            detected += len(estimated)
            hits += len(
                mir_eval.util.match_events(
                    track.onsets, estimated, config.tolerance
                )
            )
        precision = hits / max(detected, 1)
        recall = hits / max(references, 1)
        f_measure = 2 * precision * recall / max(precision + recall, 1e-9)
        if f_measure > best.f_measure:
            best = Evaluation(f_measure, precision, recall, threshold)
    return best


def fine_tune(
    initial: Path,
    bank_file: Path,
    output: Path,
    training: list[Path],
    validation: list[Path],
    seed: int,
    config: TrainConfig,
    synth: SynthConfig = CONFIG.synth,
) -> Evaluation:
    """Fine-tune with equal real/synthetic batches and real validation.

    Keep the checkpoint's feature normalization. Unlabeled real frames and
    crop edges do not contribute to the loss.
    """
    if not training or not validation:
        raise ValueError("Training and validation tracks are required.")
    if {path.resolve() for path in training} & {
        path.resolve() for path in validation
    }:
        raise ValueError("Training and validation tracks must be separate.")
    if config.batch_size < 2 or config.batch_size % 2:
        raise ValueError("Fine-tuning requires an even batch size.")
    if initial.resolve() == output.resolve() or output.exists():
        raise ValueError(f"Refusing to overwrite a checkpoint: {output}.")
    checkpoint = load_checkpoint(initial)
    if checkpoint.model.config != CONFIG.features:
        raise ValueError("Checkpoint features differ from trainer features.")
    real_training = [load_real_track(path) for path in training]
    real_validation = [load_real_track(path) for path in validation]
    logger.info(
        f"Fine-tuning {initial}: {len(training)} training tracks, "
        f"{len(validation)} validation tracks, seed {seed}, "
        f"learning rate {config.learning_rate}, 50% real examples."
    )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = checkpoint.model.to(_pick_device())
    device = next(model.parameters()).device
    loader = DataLoader(
        DropStream(bank_file, True, seed, synth),
        batch_size=config.batch_size // 2,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
    positive_weight = torch.tensor(config.positive_weight, device=device)

    def validate() -> Evaluation:
        probabilities = [
            predict(model, track.features[None], 1)[0]
            for track in real_validation
        ]
        return evaluate_real(probabilities, real_validation, config)

    best = validate()
    save_checkpoint(output, model, best.threshold)
    logger.info(
        f"Baseline validation F {best.f_measure:.4f}, "
        f"threshold {best.threshold}, saved to {output}."
    )
    running, start = 0.0, perf_counter()
    for step, (synthetic_features, synthetic_targets) in enumerate(loader, 1):
        crops = [
            _real_crop(
                real_training[int(rng.integers(len(real_training)))],
                synthetic_features.shape[-1],
                rng,
            )
            for _ in range(config.batch_size // 2)
        ]
        features = torch.cat(
            [synthetic_features, torch.stack([crop[0] for crop in crops])]
        ).to(device)
        targets = torch.cat(
            [synthetic_targets, torch.stack([crop[1] for crop in crops])]
        ).to(device)
        mask = torch.cat(
            [
                torch.ones_like(synthetic_targets),
                torch.stack([crop[2] for crop in crops]),
            ]
        ).to(device)
        mask[:, :CONTEXT_FRAMES] = 0
        mask[:, -CONTEXT_FRAMES:] = 0
        model.train()
        logits = model(features)
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=positive_weight, reduction="none"
        )
        # Normalize each example so sparse real spans keep their batch weight.
        loss = (
            (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        ).mean()
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
            if evaluation.f_measure > best.f_measure:
                best = evaluation
                save_checkpoint(output, model, best.threshold)
            logger.info(
                f"Step {step}: validation F {evaluation.f_measure:.4f}, "
                f"precision {evaluation.precision:.4f}, "
                f"recall {evaluation.recall:.4f}, "
                f"threshold {evaluation.threshold}; "
                f"best F {best.f_measure:.4f}."
            )
        if step == config.steps:
            break
    return best
