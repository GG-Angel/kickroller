"""The model input: standardized log spectrograms at three window sizes."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from analyzer.audio.io import SAMPLE_RATE


@dataclass(frozen=True)
class ModelConfig:
    sample_rate: int = SAMPLE_RATE
    hop: int = 441  # 10 ms
    windows: tuple[int, ...] = (1024, 2048, 4096)
    bands: int = 80
    fmin: float = 27.5
    fmax: float = 16000.0
    channels: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 8)
    dropout: float = 0.1

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop

    @property
    def context_frames(self) -> int:
        """Frames of context on each side of a frame."""
        return 3 + sum(self.dilations)


def mel_filterbank(config: ModelConfig, n_fft: int) -> torch.Tensor:
    """Triangular mel filters, (bins, bands). A band narrower than one bin uses its nearest bin."""

    def mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = hz(
        np.linspace(
            mel(np.array(config.fmin)), mel(np.array(config.fmax)), config.bands + 2
        )
    )
    bins = np.arange(n_fft // 2 + 1) * config.sample_rate / n_fft
    low, center, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    weights = np.maximum(
        0.0, np.minimum((bins - low) / (center - low), (high - bins) / (high - center))
    )
    for band in np.flatnonzero(weights.sum(axis=1) == 0):
        weights[band, np.argmin(np.abs(bins - center[band, 0]))] = 1.0
    return torch.tensor(weights.T, dtype=torch.float32)


class Features(nn.Module):
    """Audio (batch, samples) to standardized log spectrograms (batch, windows, bands, frames)."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        for n_fft in config.windows:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))
            self.register_buffer(f"filters_{n_fft}", mel_filterbank(config, n_fft))
        shape = (len(config.windows), config.bands, 1)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))

    def log_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        layers = []
        for n_fft in self.config.windows:
            window = getattr(self, f"window_{n_fft}")
            spectrum = torch.stft(
                audio,
                n_fft,
                hop_length=self.config.hop,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs() / (window.sum() / 2)
            bands = torch.einsum(
                "bft,fm->bmt", spectrum, getattr(self, f"filters_{n_fft}")
            )
            layers.append(torch.log1p(1000.0 * bands))
        return torch.stack(layers, dim=1)

    @torch.no_grad()
    def fit(self, audio: torch.Tensor) -> None:
        """Set the standardization from example audio (batch, samples)."""
        x = self.log_spectrogram(audio)
        self.mean.copy_(x.mean(dim=(0, 3)).unsqueeze(-1))
        self.std.copy_(x.std(dim=(0, 3)).unsqueeze(-1).clamp_min(1e-3))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return (self.log_spectrogram(audio) - self.mean) / self.std
