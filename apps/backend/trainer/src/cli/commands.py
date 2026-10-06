import typer

from src.core.logging import configure_logging

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
        raise typer.BadParameter("--verbose and --quiet cannot be used together.")
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
def synth() -> None:
    """Synthesize audio."""
    typer.echo("Synth is not implemented yet.")
