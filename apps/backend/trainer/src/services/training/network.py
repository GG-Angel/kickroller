from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from torch import nn

from src.core.config import FeatureConfig
from src.services.training.features import count_bands

CONTEXT_FRAMES = 41
"""Frames visible on each side of an output frame (±410 ms)."""
_FREQUENCY_POOL = 3
_HEAD_DILATIONS = (1, 2, 4, 8, 16)
_HEAD_CHANNELS = 64
_DROPOUT = 0.1


def _build_conv_block(
    inputs: int, outputs: int, time_kernel: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            inputs,
            outputs,
            (3, time_kernel),
            padding=(1, time_kernel // 2),
        ),
        nn.BatchNorm2d(outputs),
        nn.ReLU(),
        # Pool frequency only to keep 10 ms timing precision.
        nn.MaxPool2d((_FREQUENCY_POOL, 1)),
    )


class KickNet(nn.Module):
    """Map audio features to kick logits."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        windows, bands = len(config.windows), count_bands(config)
        self.register_buffer("mean", torch.zeros(windows, bands, 1))
        self.register_buffer("std", torch.ones(windows, bands, 1))
        self.convs = nn.Sequential(
            _build_conv_block(windows, 16, 7),
            _build_conv_block(16, 32, 7),
            _build_conv_block(32, 32, 9),
        )
        channels = 32 * (bands // _FREQUENCY_POOL**3)
        layers: list[nn.Module] = []
        # See the kick tail and its neighboring frames.
        for dilation in _HEAD_DILATIONS:
            layers += [
                nn.Conv1d(
                    channels,
                    _HEAD_CHANNELS,
                    3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.Dropout(_DROPOUT),
            ]
            channels = _HEAD_CHANNELS
        self.head = nn.Sequential(*layers, nn.Conv1d(channels, 1, 1))

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
    """Threshold with the best validation F-measure."""


def save_checkpoint(path: Path, model: KickNet, threshold: float) -> None:
    """Save the model, features, and threshold."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": model.config.model_dump(),
            "threshold": threshold,
        },
        path,
    )


def load_checkpoint(path: Path) -> Checkpoint:
    """Load a model checkpoint."""
    logger.debug(f"Loading model from {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = KickNet(FeatureConfig(**data["features"]))
    model.load_state_dict(data["state_dict"])
    checkpoint = Checkpoint(model=model.eval(), threshold=data["threshold"])
    logger.debug(f"Loaded model from {path}")
    return checkpoint
