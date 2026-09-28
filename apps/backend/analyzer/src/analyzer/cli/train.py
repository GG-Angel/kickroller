"""The train command: train the kick model."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from analyzer.cli.options import (
    MODELS,
    BankConfigFile,
    Cache,
    Quiet,
    Verbose,
    configure_logging,
)
from analyzer.model.checkpoint import DEFAULT_MODEL
from analyzer.training.bank import read_config


class Device(StrEnum):
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"


def train(
    config: BankConfigFile,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to save the model."),
    ] = DEFAULT_MODEL,
    cache: Cache = MODELS / "bank",
    steps: Annotated[
        int, typer.Option("--steps", min=1, help="Training steps.")
    ] = 15000,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, help="Drops per training step.")
    ] = 16,
    workers: Annotated[
        int,
        typer.Option("--workers", min=0, help="Processes that make training drops."),
    ] = 10,
    device: Annotated[
        Device, typer.Option("--device", help="Where to train the model.")
    ] = Device.mps,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
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
    train_model(
        bank,
        output,
        cache,
        steps=steps,
        batch_size=batch_size,
        workers=workers,
        device=device.value,
        seed=seed,
    )
