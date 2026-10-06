from pathlib import Path

import typer

from src.core.logging import configure_logging
from src.services.synthesis.bank import load_sample_bank_from_file

app = typer.Typer(
    help="Find kick onsets in a hardstyle track.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="markdown",
)


@app.callback()
def configure_cli(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only log warnings and errors."
    ),
) -> None:
    if verbose and quiet:
        raise typer.BadParameter(
            "--verbose and --quiet cannot be used together."
        )
    configure_logging(verbose=verbose, quiet=quiet)


@app.command()
def analyze() -> None:
    """Analyze audio tracks."""
    typer.echo("Analyze is not implemented yet.")


@app.command()
def train() -> None:
    """Train a kick onset model."""
    typer.echo("Train is not implemented yet.")


@app.command()
def synth(
    bank_file: Path = typer.Option(
        Path("bank.yaml"),
        "--bank-file",
        "-b",
        help="Path to the sample bank configuration file.",
    ),
) -> None:
    """Synthesize audio."""
    bank = load_sample_bank_from_file(bank_file)
    typer.echo(f"Loaded bank with {len(bank.samples)} samples.")
