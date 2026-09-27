"""A small CNN that gives a kick onset probability for each 10 ms frame.

Input: log-magnitude spectrograms at three window sizes (23, 46 and 93 ms),
in mel bands from 27.5 Hz to 16 kHz. A 2D convolution front end reads each
frame's spectrum, and a dilated temporal convolution network adds about
+/-420 ms of context (about one beat on each side at 160 BPM).
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from analyzer.audio import SAMPLE_RATE

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "kick.pt"


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


def save_model(model: KickNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state": model.state_dict()}, path)


def load_model(path: Path, device: str = "cpu") -> KickNet:
    if not path.is_file():
        raise FileNotFoundError(
            f"no kick model at {path}; train one with `analyzer train SAMPLES_DIR`"
        )
    data = torch.load(path, map_location=device, weights_only=True)
    config: dict[str, Any] = {
        k: tuple(v) if isinstance(v, list) else v for k, v in data["config"].items()
    }
    model = KickNet(ModelConfig(**config))
    model.load_state_dict(data["state"])
    return model.to(device).eval()


@torch.no_grad()
def kick_activation(
    model: KickNet, signal: np.ndarray, chunk_seconds: float = 60.0
) -> np.ndarray:
    """Kick onset probability (0-1) per frame of a mono signal; frame n is at n * hop.

    Long signals are processed in chunks, with enough overlap for the context.
    """
    config = model.config
    device = next(model.parameters()).device
    frames = len(signal) // config.hop + 1
    chunk = int(chunk_seconds * config.fps)
    margin = config.context_frames + max(config.windows) // config.hop
    out = np.zeros(frames, dtype=np.float32)
    for start in range(0, frames, chunk):
        first = max(0, start - margin)
        last = min(frames, start + chunk + margin)
        piece = signal[first * config.hop : last * config.hop]
        audio = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32))
        probability = torch.sigmoid(model(audio[None].to(device)))[0].cpu().numpy()
        stop = min(frames, start + chunk)
        out[start:stop] = probability[start - first : stop - first]
    return out
