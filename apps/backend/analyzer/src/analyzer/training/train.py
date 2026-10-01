"""Train the kick model on synthetic drops made from a sample bank."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from analyzer.detection.peaks import (
    enforce_min_distance,
    find_local_peaks,
    interpolate_peaks,
)
from analyzer.model.checkpoint import save_model
from analyzer.model.network import KickNet
from analyzer.settings import DetectorSettings, Settings
from analyzer.training.bank import BankConfig, build_bank, load_bank
from analyzer.training.synth import Catalog, Drop, make_drop


def make_targets(
    onsets: np.ndarray, frames: int, fps: float, neighbor_target: float
) -> np.ndarray:
    """1 at each onset frame and `neighbor_target` at its two neighbors."""
    target = np.zeros(frames, dtype=np.float32)
    at = np.round(onsets * fps).astype(int)
    for offset, value in ((-1, neighbor_target), (1, neighbor_target), (0, 1.0)):
        near = np.clip(at + offset, 0, frames - 1)
        target[near] = np.maximum(target[near], value)
    return target


class DropStream(IterableDataset):
    """An endless stream of (audio, targets) from training kick designs."""

    def __init__(
        self, bank_dir: Path, settings: Settings, seed: int, all_designs: bool
    ) -> None:
        self.bank_dir, self.settings, self.seed = bank_dir, settings, seed
        self.all_designs = all_designs

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        synth, model = self.settings.synth, self.settings.model
        catalog = Catalog(
            load_bank(self.bank_dir), synth.held_out_fraction, self.all_designs
        )
        while True:
            drop = make_drop(rng, catalog, synth)
            frames = len(drop.audio) // model.hop + 1
            targets = make_targets(
                drop.onsets, frames, model.fps, self.settings.training.neighbor_target
            )
            yield torch.from_numpy(drop.audio), torch.from_numpy(targets)


def count_matches(detected: np.ndarray, reference: np.ndarray, tolerance: float) -> int:
    """Onsets that match one-to-one within `tolerance` (greedy, in time order)."""
    matched, j = 0, 0
    for time in detected:
        while j < len(reference) and reference[j] < time - tolerance:
            j += 1
        if j < len(reference) and abs(reference[j] - time) <= tolerance:
            matched += 1
            j += 1
    return matched


def pick_onsets(
    probability: np.ndarray, threshold: float, fps: float, settings: DetectorSettings
) -> np.ndarray:
    """Onset times in seconds: the detector's peak picking, without the beat grid."""
    frames = find_local_peaks(
        probability, threshold, local_max_frames=round(settings.peak_window * fps)
    )
    kept = enforce_min_distance(
        frames, probability[frames], round(settings.min_distance * fps)
    )
    return interpolate_peaks(probability, frames[kept]) / fps


@torch.no_grad()
def validate(
    model: KickNet, drops: list[Drop], settings: Settings
) -> tuple[float, float, float, float]:
    """Best (F-measure, precision, recall, threshold) over the thresholds on the drops."""
    training = settings.training
    model.eval()
    probabilities = []
    for i in range(0, len(drops), training.batch_size):
        batch = drops[i : i + training.batch_size]
        audio = torch.from_numpy(np.stack([d.audio for d in batch]))
        probabilities += list(
            torch.sigmoid(model(audio.to(training.device))).cpu().numpy()
        )
    model.train()
    best = (0.0, 0.0, 0.0, training.thresholds[0])
    reference = sum(len(d.onsets) for d in drops)
    for threshold in training.thresholds:
        found = matched = 0
        for drop, probability in zip(drops, probabilities):
            onsets = pick_onsets(
                probability, threshold, model.settings.fps, settings.detector
            )
            found += len(onsets)
            matched += count_matches(onsets, drop.onsets, training.tolerance)
        precision, recall = matched / max(found, 1), matched / max(reference, 1)
        f = 2 * precision * recall / max(precision + recall, 1e-9)
        if f > best[0]:
            best = (f, precision, recall, threshold)
    return best


def train(
    bank_config: BankConfig,
    output: Path,
    bank_dir: Path,
    settings: Settings,
    all_designs: bool = False,
) -> None:
    """Train a KickNet and save the version with the best validation F-measure to `output`.

    With `all_designs`, training also uses the held-out kick designs (for a final
    model). The validation drops then have heard designs, so their F is too high.
    """
    training, synth = settings.training, settings.synth
    logger.debug("Settings: {settings}", settings=settings)
    bank = build_bank(bank_config, bank_dir, training.workers)
    catalog = Catalog(bank, synth.held_out_fraction, all_designs)
    if not (catalog.designs[False] and catalog.designs[True]):
        raise RuntimeError(
            "the bank needs more kick designs (about "
            f"{synth.held_out_fraction:.0%} are held out for validation)"
        )
    if all_designs:
        logger.warning(
            "Training on all {train} kick designs; the {held_out} validation designs "
            "are also in training, so the validation F is too high",
            train=len(catalog.designs[False]),
            held_out=len(catalog.designs[True]),
        )
    else:
        logger.info(
            "Kick designs: {train} for training, {held_out} held out for validation",
            train=len(catalog.designs[False]),
            held_out=len(catalog.designs[True]),
        )
    rng = np.random.default_rng(training.seed)
    validation = [
        make_drop(rng, catalog, synth, held_out=True)
        for _ in range(training.validation_drops)
    ]
    logger.info(
        "Made {drops} validation drops with {kicks} kicks",
        drops=len(validation),
        kicks=sum(len(d.onsets) for d in validation),
    )

    model = KickNet(settings.model).to(training.device)
    stats = np.stack(
        [make_drop(rng, catalog, synth).audio for _ in range(training.stats_drops)]
    )
    model.features.fit(torch.from_numpy(stats).to(training.device))
    logger.info(
        "Model: {params} parameters, context +/-{context} ms, training on {device}",
        params=sum(p.numel() for p in model.parameters()),
        context=round(1000 * settings.model.context_frames / settings.model.fps),
        device=training.device,
    )

    workers = training.workers
    loader = DataLoader(
        DropStream(bank_dir, settings, training.seed + 1, all_designs),
        batch_size=training.batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=training.prefetch if workers > 0 else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), training.learning_rate, weight_decay=training.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, training.learning_rate, total_steps=training.steps
    )
    steps = training.steps
    best_f, running, start = -1.0, 0.0, perf_counter()
    for step, (audio, target) in enumerate(loader, 1):
        logits = model(audio.to(training.device))
        loss = F.binary_cross_entropy_with_logits(logits, target.to(training.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        running += loss.item()
        if step % training.log_every == 0:
            logger.debug(
                "Step {step}/{steps}: loss {loss:.4f}, {rate:.1f} steps/s",
                step=step,
                steps=steps,
                loss=running / training.log_every,
                rate=step / (perf_counter() - start),
            )
            running = 0.0
        if step % training.validate_every == 0 or step == steps:
            f, precision, recall, threshold = validate(model, validation, settings)
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
