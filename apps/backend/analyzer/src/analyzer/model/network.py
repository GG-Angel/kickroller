"""The kick onset network: a 2D convolution front end and a dilated TCN."""

import torch
from torch import nn

from analyzer.model.features import Features, ModelConfig


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


def conv_block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.BatchNorm2d(outputs),
        nn.ELU(),
        nn.MaxPool2d((2, 1)),  # pool frequency only, to keep every frame
    )


class KickNet(nn.Module):
    """Kick onset logits (batch, frames) for audio (batch, samples) at `config.sample_rate`."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or ModelConfig()
        self.features = Features(config)
        self.frontend = nn.Sequential(
            conv_block(len(config.windows), 16),
            conv_block(16, 32),
            conv_block(32, 32),
        )
        self.project = nn.Conv1d(32 * (config.bands // 8), config.channels, 1)
        self.temporal = nn.Sequential(
            *(
                TemporalBlock(config.channels, d, config.dropout)
                for d in config.dilations
            )
        )
        self.head = nn.Conv1d(config.channels, 1, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.frontend(self.features(audio))
        x = self.project(x.flatten(1, 2))
        return self.head(self.temporal(x)).squeeze(1)
