from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.services.synthesis import bank


def test_load_sample_bank_resolves_workspace_paths_and_globs(tmp_path: Path) -> None:
    direct_path = tmp_path / "direct.wav"
    globbed_path = tmp_path / "loops" / "loop.wav"
    direct_path.touch()
    globbed_path.parent.mkdir()
    globbed_path.touch()
    config = bank.SampleBankConfig(
        workspace=tmp_path,
        samples=[
            bank.SampleConfig(
                kind="kick",
                paths=[Path("direct.wav"), Path("loops/*.wav")],
            )
        ],
    )

    with patch(
        "src.services.synthesis.bank.load_audio", return_value=np.zeros(0)
    ) as load_audio:
        sample_bank = bank.load_sample_bank(config)

    assert [call.args[0] for call in load_audio.call_args_list] == [
        direct_path,
        globbed_path,
    ]
    assert [sample.path for sample in sample_bank.samples] == [
        direct_path,
        globbed_path,
    ]


def test_load_sample_bank_raises_when_path_does_not_match(tmp_path: Path) -> None:
    config = bank.SampleBankConfig(
        workspace=tmp_path,
        samples=[bank.SampleConfig(kind="kick", paths=[Path("missing.wav")])],
    )

    with pytest.raises(FileNotFoundError, match="No files matched"):
        bank.load_sample_bank(config)
