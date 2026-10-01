"""Save and load trained models, and run a model on a signal."""

from pathlib import Path

import numpy as np
import torch

from analyzer.model.network import KickNet
from analyzer.settings import ModelSettings

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "kick.pt"


def save_model(model: KickNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.settings.model_dump(), "state": model.state_dict()}, path
    )


def load_model(path: Path, device: str = "cpu") -> KickNet:
    """A trained model, built with the model settings saved in its file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no kick model at {path}; train one with `analyzer train bank.toml`"
        )
    data = torch.load(path, map_location=device, weights_only=True)
    # Older models also saved settings that are now constants (the sample rate).
    saved = {k: v for k, v in data["config"].items() if k in ModelSettings.model_fields}
    model = KickNet(ModelSettings.model_validate(saved))
    model.load_state_dict(data["state"])
    return model.to(device).eval()


@torch.no_grad()
def predict_kick_probability(
    model: KickNet, signal: np.ndarray, chunk_seconds: float
) -> np.ndarray:
    """Kick onset probability (0-1) per frame of a mono signal; frame n is at n * hop.

    Long signals are processed in chunks, with enough overlap for the context.
    """
    settings = model.settings
    device = next(model.parameters()).device
    frames = len(signal) // settings.hop + 1
    chunk = int(chunk_seconds * settings.fps)
    margin = settings.context_frames + max(settings.windows) // settings.hop
    out = np.zeros(frames, dtype=np.float32)
    for start in range(0, frames, chunk):
        first = max(0, start - margin)
        last = min(frames, start + chunk + margin)
        piece = signal[first * settings.hop : last * settings.hop]
        audio = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32))
        probability = torch.sigmoid(model(audio[None].to(device)))[0].cpu().numpy()
        stop = min(frames, start + chunk)
        out[start:stop] = probability[start - first : stop - first]
    return out
