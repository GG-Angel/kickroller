"""Train the kick model on synthetic drops made from the On Point Samples packs."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from analyzer.bank import build_bank, load_bank
from analyzer.model import KickNet, ModelConfig, save_model
from analyzer.peaks import enforce_min_distance, local_peaks
from analyzer.synth import Catalog, Drop, make_drop

VALIDATION_DROPS = 200
STATS_DROPS = 64
TOLERANCE = 0.02  # onset match window, in seconds
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def targets(onsets: np.ndarray, frames: int, config: ModelConfig) -> np.ndarray:
    """1 at each onset frame and 0.5 at its neighbors."""
    target = np.zeros(frames, dtype=np.float32)
    at = np.round(onsets * config.fps).astype(int)
    for offset, value in ((-1, 0.5), (1, 0.5), (0, 1.0)):
        near = np.clip(at + offset, 0, frames - 1)
        target[near] = np.maximum(target[near], value)
    return target


class DropStream(IterableDataset):
    """An endless stream of (audio, targets) from training kick designs."""

    def __init__(self, bank_dir: Path, config: ModelConfig, seed: int) -> None:
        self.bank_dir, self.config, self.seed = bank_dir, config, seed

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        catalog = Catalog(load_bank(self.bank_dir))
        while True:
            drop = make_drop(rng, catalog)
            frames = len(drop.audio) // self.config.hop + 1
            yield (
                torch.from_numpy(drop.audio),
                torch.from_numpy(targets(drop.onsets, frames, self.config)),
            )


def match_count(detected: np.ndarray, reference: np.ndarray, tolerance: float) -> int:
    """Onsets that match one-to-one within `tolerance` (greedy, in time order)."""
    matched, j = 0, 0
    for time in detected:
        while j < len(reference) and reference[j] < time - tolerance:
            j += 1
        if j < len(reference) and abs(reference[j] - time) <= tolerance:
            matched += 1
            j += 1
    return matched


def pick_onsets(probability: np.ndarray, threshold: float, fps: float) -> np.ndarray:
    frames = local_peaks(probability, threshold, local_max_frames=round(0.02 * fps))
    kept = enforce_min_distance(frames, probability[frames], round(0.04 * fps))
    return frames[kept] / fps


@torch.no_grad()
def validate(
    model: KickNet, drops: list[Drop], device: str, batch_size: int
) -> tuple[float, float, float, float]:
    """Best (F-measure, precision, recall, threshold) over THRESHOLDS on the drops."""
    model.eval()
    probabilities = []
    for i in range(0, len(drops), batch_size):
        audio = torch.from_numpy(np.stack([d.audio for d in drops[i : i + batch_size]]))
        probabilities += list(torch.sigmoid(model(audio.to(device))).cpu().numpy())
    model.train()
    best = (0.0, 0.0, 0.0, THRESHOLDS[0])
    reference = sum(len(d.onsets) for d in drops)
    for threshold in THRESHOLDS:
        found = matched = 0
        for drop, probability in zip(drops, probabilities):
            onsets = pick_onsets(probability, threshold, model.config.fps)
            found += len(onsets)
            matched += match_count(onsets, drop.onsets, TOLERANCE)
        precision, recall = matched / max(found, 1), matched / max(reference, 1)
        f = 2 * precision * recall / max(precision + recall, 1e-9)
        if f > best[0]:
            best = (f, precision, recall, threshold)
    return best


def train(
    samples: Path,
    output: Path,
    bank_dir: Path,
    steps: int = 15000,
    batch_size: int = 16,
    workers: int = 10,
    learning_rate: float = 2e-3,
    validate_every: int = 500,
    device: str = "mps",
    seed: int = 0,
) -> None:
    """Train a KickNet and save the version with the best validation F-measure to `output`."""
    bank = build_bank(samples, bank_dir, workers)
    catalog = Catalog(bank)
    logger.info(
        "Kick designs: {train} for training, {held_out} held out for validation",
        train=len(catalog.designs[False]),
        held_out=len(catalog.designs[True]),
    )
    rng = np.random.default_rng(seed)
    validation = [
        make_drop(rng, catalog, held_out=True) for _ in range(VALIDATION_DROPS)
    ]
    logger.info(
        "Made {drops} validation drops with {kicks} kicks",
        drops=len(validation),
        kicks=sum(len(d.onsets) for d in validation),
    )

    config = ModelConfig()
    model = KickNet(config).to(device)
    stats = np.stack([make_drop(rng, catalog).audio for _ in range(STATS_DROPS)])
    model.features.fit(torch.from_numpy(stats).to(device))
    logger.info(
        "Model: {params} parameters, context +/-{context} ms, training on {device}",
        params=sum(p.numel() for p in model.parameters()),
        context=round(1000 * config.context_frames / config.fps),
        device=device,
    )

    loader = DataLoader(
        DropStream(bank_dir, config, seed + 1),
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, learning_rate, total_steps=steps
    )
    best_f, running, start = -1.0, 0.0, perf_counter()
    for step, (audio, target) in enumerate(loader, 1):
        logits = model(audio.to(device))
        loss = F.binary_cross_entropy_with_logits(logits, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        running += loss.item()
        if step % 100 == 0:
            logger.debug(
                "Step {step}/{steps}: loss {loss:.4f}, {rate:.1f} steps/s",
                step=step,
                steps=steps,
                loss=running / 100,
                rate=step / (perf_counter() - start),
            )
            running = 0.0
        if step % validate_every == 0 or step == steps:
            f, precision, recall, threshold = validate(
                model, validation, device, batch_size
            )
            improved = f > best_f
            if improved:
                best_f = f
                save_model(model, output)
            logger.info(
                "Step {step}/{steps}: validation F {f:.3f} (precision {precision:.3f}, "
                "recall {recall:.3f} at threshold {threshold}){saved}",
                step=step,
                steps=steps,
                f=f,
                precision=precision,
                recall=recall,
                threshold=threshold,
                saved=f", saved to {output}" if improved else "",
            )
        if step == steps:
            break
    logger.info(
        "Training done in {minutes:.0f} min; best validation F {f:.3f}",
        minutes=(perf_counter() - start) / 60,
        f=best_f,
    )
