from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from src.cli.commands import app
from src.models.audio import LabeledDrop
from src.services.synthesis.bank import SampleBank


def test_synth_can_bypass_sample_bank_cache(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.yaml"
    bank_path.touch()
    drop = LabeledDrop(
        signal=np.zeros(10, dtype=np.float32), onsets=np.zeros(0)
    )

    with (
        patch(
            "src.cli.commands.load_sample_bank_from_file",
            return_value=SampleBank(samples=[]),
        ) as load_bank,
        patch("src.cli.commands.generate_drop", return_value=drop),
        patch("src.cli.commands.save_audio"),
        patch("src.cli.commands.save_onsets"),
    ):
        result = CliRunner().invoke(
            app,
            ["synth", "--bank-file", str(bank_path), "--no-cache"],
        )

    assert result.exit_code == 0
    load_bank.assert_called_once_with(bank_path, use_cache=False)
