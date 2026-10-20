from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.core.config import CONFIG, FeatureConfig
from src.services.training.features import band_count

CONTEXT_FRAMES = 15
"""Frames on each side of a frame that its output can see (±150 ms)."""
_FREQUENCY_POOL = 3


def _conv_block(inputs: int, outputs: int, time_kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            inputs,
            outputs,
            (3, time_kernel),
            padding=(1, time_kernel // 2),
        ),
        nn.BatchNorm2d(outputs),
        nn.ReLU(),
        # pool frequency only: pooling time would lose the 10 ms precision
        nn.MaxPool2d((_FREQUENCY_POOL, 1)),
    )


class KickNet(nn.Module):
    """Features (batch, windows, bands, frames) to kick logits (batch, frames)."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, config: FeatureConfig = CONFIG.features) -> None:
        super().__init__()
        self.config = config
        windows, bands = len(config.windows), band_count(config)
        self.register_buffer("mean", torch.zeros(windows, bands, 1))
        self.register_buffer("std", torch.ones(windows, bands, 1))
        self.convs = nn.Sequential(
            _conv_block(windows, 16, 7),
            _conv_block(16, 32, 7),
            _conv_block(32, 32, 9),
        )
        pooled_bands = bands // _FREQUENCY_POOL**3
        self.head = nn.Sequential(
            nn.Conv1d(32 * pooled_bands, 64, 11, padding=5),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
        )

    @torch.no_grad()
    def set_standardization(self, features: torch.Tensor) -> None:
        """Set the mean and std of each band from training features."""
        self.mean.copy_(features.mean(dim=(0, 3)).unsqueeze(-1))
        self.std.copy_(features.std(dim=(0, 3)).unsqueeze(-1).clamp_min(1e-3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.convs((features - self.mean) / self.std)
        return self.head(x.flatten(1, 2)).squeeze(1)


@dataclass(frozen=True)
class Checkpoint:
    model: KickNet
    threshold: float
    """Peak threshold with the best validation F-measure."""


def save_checkpoint(path: Path, model: KickNet, threshold: float) -> None:
    """Save a checkpoint containing the model state and threshold."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": model.config.model_dump(),
            "threshold": threshold,
        },
        path,
    )


def load_checkpoint(path: Path) -> Checkpoint:
    """Load a checkpoint containing the model state and threshold."""
    data = torch.load(path, map_location="cpu")
    model = KickNet(FeatureConfig(**data["features"]))
    model.load_state_dict(data["state_dict"])
    return Checkpoint(model=model.eval(), threshold=data["threshold"])
