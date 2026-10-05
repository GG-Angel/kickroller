from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from core.logging import configure_logging
from services.synthesizer.bank import load_bank_from_file
from services.synthesizer.export import export_drops

app = typer.Typer(
    help="Detects kick onsets in a hardstyle track.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="markdown",
)


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show debug logs."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only show warnings and errors."),
    ] = False,
) -> None:
    configure_logging(verbose=verbose, quiet=quiet)


@app.command(name="analyze")
def analyze() -> None:
    """Detect kick onsets and the beat grid in a hardstyle track."""
    logger.info("erm")


@app.command(name="train")
def train() -> None:
    """Train the kick onset detection model."""
    logger.info("erm")


@app.command(name="synth")
def synth(
    bank_path: Annotated[
        Path,
        typer.Option("--bank", "-b", help="Path to the sample bank configuration."),
    ] = Path("bank.toml"),
    cache: Annotated[
        Path,
        typer.Option(
            "--cache",
            file_okay=False,
            help="Cache folder for the decoded samples (made on the first run).",
        ),
    ] = Path("models/bank"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Output folder."),
    ] = Path("models/drops"),
    count: Annotated[
        int, typer.Option("--count", "-n", min=1, help="Number of drops.")
    ] = 4,
    seed: Annotated[
        int | None,
        typer.Option(min=0, help="Random seed (default: a new seed in the log)."),
    ] = None,
    clicks: Annotated[
        bool,
        typer.Option("--clicks", help="Also write each drop with its kick clicks."),
    ] = False,
    held_out: Annotated[
        bool,
        typer.Option("--held-out", help="Use only the held-out kicks."),
    ] = False,
) -> None:
    """Write synthetic drops with labeled kick onsets, to check the training data."""
    bank = load_bank_from_file(path=bank_path, cache=cache)
    export_drops(
        bank,
        folder=output,
        num_drops=count,
        seed=seed,
        include_clicks=clicks,
        held_out=held_out,
    )
