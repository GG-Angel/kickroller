from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from src.cli.commands import app
from src.core.config import CONFIG
from src.models.audio import SampleKind
from src.services.synthesis.bank import SampleBank


def test_synth_can_bypass_sample_bank_cache(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank.yaml"
    bank_path.touch()

    with (
        patch(
            "src.cli.commands.load_sample_bank_from_file",
            return_value=SampleBank(samples=[]),
        ) as load_bank,
        patch("src.cli.commands.export_drops"),
    ):
        result = CliRunner().invoke(
            app,
            ["synth", "--bank-file", str(bank_path), "--no-cache"],
        )

    assert result.exit_code == 0
    load_bank.assert_called_once_with(bank_path, use_cache=False)


def test_synth_writes_drops_to_output_folder(tmp_path: Path) -> None:
    bank = SampleBank(samples=[])
    output = tmp_path / "out"

    with (
        patch("src.cli.commands.load_sample_bank_from_file", return_value=bank),
        patch("src.cli.commands.export_drops") as export,
    ):
        result = CliRunner().invoke(
            app, ["synth", "-o", str(output), "-n", "3", "--seed", "5"]
        )

    assert result.exit_code == 0
    export.assert_called_once_with(bank, output, 3, 5, CONFIG.synth)


def _invoke_synth(*options: str) -> list:
    with (
        patch(
            "src.cli.commands.load_sample_bank_from_file",
            return_value=SampleBank(samples=[]),
        ),
        patch("src.cli.commands.export_drops") as export,
    ):
        result = CliRunner().invoke(app, ["synth", *options])

    assert result.exit_code == 0, result.output
    return export.call_args.args[4].layers


def test_synth_uses_all_layers_by_default() -> None:
    assert _invoke_synth() == CONFIG.synth.layers


def test_synth_can_use_only_some_kinds() -> None:
    layers = _invoke_synth("-k", "kick", "--kind", "drum")

    assert layers
    assert {layer.kind for layer in layers} == {SampleKind.DRUM}


def test_synth_can_use_kicks_only() -> None:
    assert _invoke_synth("-k", "kick") == []


def test_synth_rejects_unknown_kinds() -> None:
    result = CliRunner().invoke(app, ["synth", "-k", "bass"])

    assert result.exit_code != 0
