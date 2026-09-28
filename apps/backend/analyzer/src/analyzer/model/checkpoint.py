"""Save and load trained models, and run a model on a signal."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyzer.model.features import ModelConfig
from analyzer.model.network import KickNet

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "kick.pt"


def save_model(model: KickNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state": model.state_dict()}, path)


def load_model(path: Path, device: str = "cpu") -> KickNet:
    if not path.is_file():
        raise FileNotFoundError(
            f"no kick model at {path}; train one with `analyzer train bank.toml`"
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
