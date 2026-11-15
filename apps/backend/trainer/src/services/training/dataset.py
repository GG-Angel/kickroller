from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from src.core.config import FeatureConfig, SynthConfig
from src.models.audio import LabeledDrop
from src.services.synthesis.bank import load_sample_bank_from_file
from src.services.synthesis.synth import generate_cropped_drop
from src.services.training.features import compute_features, make_targets
from torch.utils.data import IterableDataset, get_worker_info


class Example(NamedTuple):
    features: torch.Tensor
    targets: torch.Tensor


def make_example(drop: LabeledDrop, config: FeatureConfig, sr: int) -> Example:
    features = compute_features(drop.signal, config, sr)
    targets = make_targets(drop.onsets, features.shape[-1], config, sr)
    return Example(torch.from_numpy(features), torch.from_numpy(targets))


class DropStream(IterableDataset[Example]):
    """Yield endless examples from new drops with training kicks."""

    def __init__(
        self,
        bank_file: Path,
        use_cache: bool,
        seed: int,
        config: SynthConfig,
        feature_config: FeatureConfig,
        sr: int,
    ) -> None:
        # Load the bank in each worker to avoid copying its memory maps.
        self.bank_file = bank_file
        self.use_cache = use_cache
        self.seed = seed
        self.config = config
        self.feature_config = feature_config
        self.sr = sr

    def __iter__(self) -> Iterator[Example]:
        worker = get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0])
        bank = load_sample_bank_from_file(
            path=self.bank_file,
            sr=self.sr,
            use_cache=self.use_cache,
        ).select_kicks(
            held_out=False,
            fraction=self.config.held_out_kick_fraction,
        )
        while True:
            yield make_example(
                generate_cropped_drop(bank, rng, self.config, self.sr),
                self.feature_config,
                self.sr,
            )
