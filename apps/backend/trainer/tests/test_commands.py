from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from src.cli.commands import app
from src.core.config import CONFIG
from src.models.audio import SampleKind
from src.services.analysis.analyze import Analysis
from src.services.synthesis.bank import Sample, SampleBank, is_held_out


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


@pytest.mark.parametrize("held_out", [False, True])
def test_synth_uses_only_the_kicks_of_the_split(held_out: bool) -> None:
    kicks = [
        Sample(
            name=f"kick_{index}.wav",
            path=Path(f"kick_{index}.wav"),
            signal=np.zeros(10, dtype=np.float32),
            kind=SampleKind.KICK,
            bpm=160,
            is_loop=False,
        )
        for index in range(100)
    ]

    with (
        patch(
            "src.cli.commands.load_sample_bank_from_file",
            return_value=SampleBank(samples=kicks),
        ),
        patch("src.cli.commands.export_drops") as export,
    ):
        options = ["--held-out"] if held_out else []
        result = CliRunner().invoke(app, ["synth", *options])

    assert result.exit_code == 0, result.output
    used = export.call_args.args[0].samples
    fraction = CONFIG.synth.held_out_kick_fraction
    assert used
    assert all(is_held_out(kick.name, fraction) == held_out for kick in used)


@pytest.mark.parametrize("options", [[], ["--no-master"]])
def test_synth_can_turn_off_the_master_effects(options: list[str]) -> None:
    with (
        patch(
            "src.cli.commands.load_sample_bank_from_file",
            return_value=SampleBank(samples=[]),
        ),
        patch("src.cli.commands.export_drops") as export,
    ):
        result = CliRunner().invoke(app, ["synth", "-k", "kick", *options])

    assert result.exit_code == 0, result.output
    config = export.call_args.args[4]
    assert config.layers == []
    if options:
        assert config.master is None
    else:
        assert config.master == CONFIG.synth.master


def test_train_passes_the_options(tmp_path: Path) -> None:
    output = tmp_path / "model.pt"
    with patch("src.cli.commands.train_model") as train:
        train.return_value.f_measure = 0.9
        train.return_value.threshold = 0.5
        result = CliRunner().invoke(
            app,
            ["train", "-o", str(output), "--steps", "7", "--seed", "3"],
        )

    assert result.exit_code == 0, result.output
    bank_file, saved, use_cache, seed, config = train.call_args.args
    assert (bank_file, saved, use_cache, seed) == (
        Path("bank.yaml"),
        output,
        True,
        3,
    )
    assert config.steps == 7


def test_analyze_writes_onsets_clicks_and_scores(tmp_path: Path) -> None:
    track = tmp_path / "track.wav"
    model = tmp_path / "model.pt"
    labels = tmp_path / "labels.csv"
    for path in (track, model):
        path.touch()
    labels.write_text("0.5\n1.0\n")
    analysis = Analysis(onsets=np.array([0.5, 1.0]), confidences=np.ones(2))

    with (
        patch("src.cli.commands.load_checkpoint") as load_model,
        patch(
            "src.cli.commands.load_audio",
            return_value=np.zeros(CONFIG.sr, dtype=np.float32),
        ),
        patch("src.cli.commands.analyze_signal", return_value=analysis) as run,
    ):
        load_model.return_value.threshold = 0.85
        result = CliRunner().invoke(
            app,
            [
                "analyze",
                str(track),
                "-m",
                str(model),
                "--threshold",
                "0.3",
                "--clicks",
                "--labels",
                str(labels),
            ],
        )

    assert result.exit_code == 0, result.output
    assert run.call_args.args[2] == 0.3
    assert (tmp_path / "track.csv").exists()
    assert (tmp_path / "track.clicks.wav").exists()
    assert "F-measure 1.000" in result.output


def test_analyze_can_write_onsets_without_confidences(tmp_path: Path) -> None:
    track = tmp_path / "track.wav"
    model = tmp_path / "model.pt"
    for path in (track, model):
        path.touch()
    analysis = Analysis(onsets=np.array([0.5, 1.0]), confidences=np.ones(2))

    with (
        patch("src.cli.commands.load_checkpoint"),
        patch(
            "src.cli.commands.load_audio",
            return_value=np.zeros(CONFIG.sr, dtype=np.float32),
        ),
        patch("src.cli.commands.analyze_signal", return_value=analysis),
    ):
        result = CliRunner().invoke(
            app, ["analyze", str(track), "-m", str(model), "--no-confidences"]
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "track.csv").read_text() == "0.500000\n1.000000\n"
