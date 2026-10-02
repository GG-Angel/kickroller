"""The model input: standardized log spectrograms at three window sizes."""

import numpy as np
import torch
from torch import nn

from analyzer.audio.io import SAMPLE_RATE
from analyzer.settings import ModelSettings

# The HTK mel scale: mel = MEL_FACTOR * log10(1 + hz / MEL_BREAK_HZ).
MEL_FACTOR = 2595.0
MEL_BREAK_HZ = 700.0
MIN_STD = 1e-3  # the lowest standardization divisor, for near-constant bands


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return MEL_FACTOR * np.log10(1.0 + hz / MEL_BREAK_HZ)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return MEL_BREAK_HZ * (10.0 ** (mel / MEL_FACTOR) - 1.0)


def _build_mel_filterbank(settings: ModelSettings, n_fft: int) -> torch.Tensor:
    """Triangular mel filters, (bins, bands). A band narrower than one bin uses its nearest bin."""
    edges = _mel_to_hz(
        np.linspace(
            _hz_to_mel(np.array(settings.fmin)),
            _hz_to_mel(np.array(settings.fmax)),
            settings.bands + 2,
        )
    )
    bins = np.arange(n_fft // 2 + 1) * SAMPLE_RATE / n_fft
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

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.settings = settings
        for n_fft in settings.windows:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft))
            self.register_buffer(
                f"filters_{n_fft}", _build_mel_filterbank(settings, n_fft)
            )
        shape = (len(settings.windows), settings.bands, 1)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))

    def _compute_log_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        layers = []
        for n_fft in self.settings.windows:
            window = getattr(self, f"window_{n_fft}")
            # Divided by half the window sum, a full-scale sine has magnitude 1.
            spectrum = torch.stft(
                audio,
                n_fft,
                hop_length=self.settings.hop,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            ).abs() / (window.sum() / 2)
            bands = torch.einsum(
                "bft,fm->bmt", spectrum, getattr(self, f"filters_{n_fft}")
            )
            layers.append(torch.log1p(self.settings.log_compression * bands))
        return torch.stack(layers, dim=1)

    @torch.no_grad()
    def fit(self, audio: torch.Tensor) -> None:
        """Set the standardization from example audio (batch, samples)."""
        x = self._compute_log_spectrogram(audio)
        self.mean.copy_(x.mean(dim=(0, 3)).unsqueeze(-1))
        self.std.copy_(x.std(dim=(0, 3)).unsqueeze(-1).clamp_min(MIN_STD))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return (self._compute_log_spectrogram(audio) - self.mean) / self.std
