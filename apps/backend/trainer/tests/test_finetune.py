from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.core.config import CONFIG
from src.services.training.finetune import (
    RealTrack,
    _real_crop,
    evaluate_real,
    fine_tune,
    load_real_track,
)
from src.services.training.network import KickNet, save_checkpoint


def test_load_real_track_masks_unlabeled_gaps(tmp_path: Path) -> None:
    path = tmp_path / "track.mp3"
    path.with_suffix(".csv").write_text("1.0\n1.5\n10.0\n10.5\n")
    with patch(
        "src.services.training.finetune.load_audio",
        return_value=np.zeros(CONFIG.sr * 12, dtype=np.float32),
    ):
        track = load_real_track(path, CONFIG.features, CONFIG.train, CONFIG.sr)

    assert track.mask[100] == track.mask[150] == track.mask[1000] == 1
    assert track.mask[0] == track.mask[500] == track.mask[1190] == 0
    assert track.targets[100] == 1


@pytest.mark.parametrize("labels", ["", "2\n1\n", "-1\n", "nan\n", "20\n"])
def test_load_real_track_rejects_invalid_labels(
    tmp_path: Path, labels: str
) -> None:
    path = tmp_path / "track.mp3"
    path.with_suffix(".csv").write_text(labels)
    with (
        patch(
            "src.services.training.finetune.load_audio",
            return_value=np.zeros(CONFIG.sr * 12, dtype=np.float32),
        ),
        pytest.raises(ValueError, match="Invalid or empty"),
    ):
        load_real_track(path, CONFIG.features, CONFIG.train, CONFIG.sr)


def _track() -> RealTrack:
    features = torch.zeros(3, 109, 2000)
    mask = torch.zeros(2000)
    mask[500:700] = 1
    return RealTrack(features, torch.zeros(2000), mask, np.array([5.0, 6.0]))


def test_real_crops_keep_scored_frames() -> None:
    rng = np.random.default_rng(0)
    track = _track()
    for _ in range(10):
        features, targets, mask = _real_crop(track, 1201, rng)
        assert features.shape == (3, 109, 1201)
        assert targets.shape == mask.shape == (1201,)
        assert mask[41:-41].sum() > 0


def test_real_validation_ignores_predictions_between_spans() -> None:
    track = _track()
    track = RealTrack(
        track.features, track.targets, track.mask, np.array([1.0, 10.0])
    )
    probabilities = np.zeros(1201, dtype=np.float32)
    probabilities[[100, 500, 1000]] = 0.9

    best = evaluate_real(
        [probabilities],
        [track],
        CONFIG.train,
        CONFIG.features,
        CONFIG.sr,
    )

    assert best.f_measure == best.precision == best.recall == 1.0


def test_fine_tune_rejects_overlapping_tracks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be separate"):
        fine_tune(
            tmp_path / "base.pt",
            tmp_path / "bank.yaml",
            tmp_path / "out.pt",
            [tmp_path / "track.mp3"],
            [tmp_path / "track.mp3"],
            1,
            CONFIG.train,
            CONFIG.synth,
            CONFIG.features,
            CONFIG.sr,
        )


def test_fine_tune_keeps_best_checkpoint_and_normalization(
    tmp_path: Path,
) -> None:
    initial, output = tmp_path / "base.pt", tmp_path / "out.pt"
    model = KickNet(CONFIG.features)
    model.mean.fill_(3.0)
    save_checkpoint(initial, model, 0.8)
    track = _track()
    config = CONFIG.train.model_copy(
        update={"steps": 1, "batch_size": 2, "workers": 0}
    )
    batch = (torch.zeros(1, 3, 109, 1201), torch.zeros(1, 1201))
    with (
        patch(
            "src.services.training.finetune.load_real_track",
            return_value=track,
        ),
        patch(
            "src.services.training.finetune.DataLoader",
            return_value=[batch],
        ),
        patch(
            "src.services.training.loop._pick_device",
            return_value=torch.device("cpu"),
        ),
    ):
        best = fine_tune(
            initial,
            tmp_path / "bank.yaml",
            output,
            [tmp_path / "train.mp3"],
            [tmp_path / "validate.mp3"],
            1,
            config,
            CONFIG.synth,
            CONFIG.features,
            CONFIG.sr,
        )
    saved = torch.load(output, weights_only=True)
    assert saved["threshold"] == best.threshold
    assert torch.all(saved["state_dict"]["mean"] == 3)
    assert initial.read_bytes() != b""
