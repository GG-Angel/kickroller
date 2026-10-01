"""The train command: train the kick model."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from analyzer.cli.options import (
    MODELS,
    SETTINGS,
    BankConfigFile,
    Cache,
    Quiet,
    Verbose,
    configure_logging,
)
from analyzer.model.checkpoint import DEFAULT_MODEL
from analyzer.training.bank import read_config

DEFAULTS = SETTINGS.training


class Device(StrEnum):
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"


DEFAULT_DEVICE = Device(DEFAULTS.device)


def train(
    config: BankConfigFile,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to save the model."),
    ] = DEFAULT_MODEL,
    cache: Cache = MODELS / "bank",
    steps: Annotated[
        int, typer.Option("--steps", min=1, help="Training steps.")
    ] = DEFAULTS.steps,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Drops per training step.")
    ] = DEFAULTS.batch_size,
    workers: Annotated[
        int,
        typer.Option("--workers", min=0, help="Processes that make training drops."),
    ] = DEFAULTS.workers,
    device: Annotated[
        Device, typer.Option("--device", help="Where to train the model.")
    ] = DEFAULT_DEVICE,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = DEFAULTS.seed,
    all_designs: Annotated[
        bool,
        typer.Option(
            "--all-designs",
            help="Also train on the held-out kick designs, for a final model after "
            "tuning. The validation drops then have heard designs, so their F is "
            "too high.",
        ),
    ] = False,
    verbose: Verbose = 0,
    quiet: Quiet = False,
) -> None:
    """Train the kick model on synthetic drops made from the sample bank."""
    from analyzer.training.train import train as train_model

    configure_logging(verbose, quiet)
    try:
        bank = read_config(config)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="CONFIG") from error
    training = DEFAULTS.model_copy(
        update={
            "steps": steps,
            "batch_size": batch_size,
            "workers": workers,
            "device": device.value,
            "seed": seed,
        }
    )
    settings = SETTINGS.model_copy(update={"training": training})
    train_model(bank, output, cache, settings, all_designs)
