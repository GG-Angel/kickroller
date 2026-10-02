from typing import Annotated

import typer

from core.logging import configure_logging

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
    pass


@app.command(name="train")
def train() -> None:
    """Train the kick onset detection model."""
    pass


@app.command(name="generate")
def generate() -> None:
    """Generate a synthetic drop with labeled kick onsets."""
    pass
