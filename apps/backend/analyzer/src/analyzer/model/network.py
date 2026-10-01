"""The kick onset network: a 2D convolution front end and a dilated TCN."""

from itertools import pairwise

import torch
from torch import nn

from analyzer.model.features import Features
from analyzer.settings import ModelSettings

KERNEL_SIZE = 3  # of all convolutions (ModelSettings.context_frames uses this)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels, channels, KERNEL_SIZE, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


def make_conv_block(inputs: int, outputs: int) -> nn.Sequential:
    """A 2D convolution layer that halves the frequency bands and keeps every frame."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
        nn.BatchNorm2d(outputs),
        nn.ELU(),
        nn.MaxPool2d((2, 1)),
    )


class KickNet(nn.Module):
    """Kick onset logits (batch, frames) for audio (batch, samples) at SAMPLE_RATE."""

    def __init__(self, settings: ModelSettings | None = None) -> None:
        super().__init__()
        self.settings = settings = settings or ModelSettings()
        self.features = Features(settings)
        channels = (len(settings.windows), *settings.frontend_channels)
        self.frontend = nn.Sequential(
            *(make_conv_block(a, b) for a, b in pairwise(channels))
        )
        bands = settings.bands // 2 ** len(settings.frontend_channels)
        self.project = nn.Conv1d(channels[-1] * bands, settings.channels, 1)
        self.temporal = nn.Sequential(
            *(
                TemporalBlock(settings.channels, d, settings.dropout)
                for d in settings.dilations
            )
        )
        self.head = nn.Conv1d(settings.channels, 1, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.frontend(self.features(audio))
        x = self.project(x.flatten(1, 2))
        return self.head(self.temporal(x)).squeeze(1)
