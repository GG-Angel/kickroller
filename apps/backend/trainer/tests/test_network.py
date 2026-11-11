from pathlib import Path

import torch

from src.core.config import CONFIG
from src.services.training.features import count_bands
from src.services.training.network import (
    CONTEXT_FRAMES,
    KickNet,
    load_checkpoint,
    save_checkpoint,
)


def _make_features(frames: int = 100) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 3, count_bands(CONFIG.features), frames)


def test_kick_net_gives_one_logit_per_frame() -> None:
    assert KickNet(CONFIG.features)(_make_features()).shape == (2, 100)


def test_kick_net_sees_exactly_its_context() -> None:
    model = KickNet(CONFIG.features).eval()
    features = _make_features()
    changed = features.clone()
    changed[:, :, :, 50] += 5.0

    with torch.no_grad():
        difference = (model(changed) - model(features)).abs().sum(dim=0)

    seen = torch.nonzero(difference > 1e-6).flatten()
    assert seen.min() == 50 - CONTEXT_FRAMES
    assert seen.max() == 50 + CONTEXT_FRAMES


def test_kick_net_standardizes_its_input() -> None:
    model = KickNet(CONFIG.features)
    features = _make_features() * 3 + 7

    model.set_standardization(features)

    standardized = (features - model.mean) / model.std
    assert standardized.mean().abs() < 1e-4
    assert (standardized.std() - 1).abs() < 1e-2


def test_checkpoint_keeps_the_model_and_threshold(tmp_path: Path) -> None:
    model = KickNet(CONFIG.features)
    model.set_standardization(_make_features() + 2)
    model.eval()
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, threshold=0.35)
    checkpoint = load_checkpoint(path)

    assert checkpoint.threshold == 0.35
    with torch.no_grad():
        torch.testing.assert_close(
            checkpoint.model(_make_features()), model(_make_features())
        )
