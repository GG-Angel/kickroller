"""Options, settings and logging shared by the commands."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError

from analyzer.model.checkpoint import DEFAULT_MODEL
from analyzer.settings import Settings

MODELS = DEFAULT_MODEL.parent

LOG_LEVELS = ("INFO", "DEBUG", "TRACE")
LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)

Verbose = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        count=True,
        help="Show more logs: -v for the pipeline steps, "
        "-vv also for each candidate kick.",
    ),
]
Quiet = Annotated[
    bool,
    typer.Option("--quiet", "-q", help="Only show warnings and errors."),
]
BankConfigFile = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        help="The sample bank config (TOML, see bank.example.toml).",
    ),
]
Cache = Annotated[
    Path,
    typer.Option(
        "--cache",
        help="Cache folder for the decoded samples (made on the first run).",
    ),
]


def _load_settings() -> Settings:
    """The settings, or exit with the error if an ANALYZER_* value is not valid."""
    try:
        return Settings()
    except ValidationError as error:
        typer.echo(f"Settings error (ANALYZER_* or .env): {error}", err=True)
        raise SystemExit(2) from error


# The CLI option defaults come from these settings.
SETTINGS = _load_settings()


def configure_logging(verbose: int, quiet: bool) -> None:
    """Log to standard error, so that standard output only has the result."""
    level = "WARNING" if quiet else LOG_LEVELS[min(verbose, len(LOG_LEVELS) - 1)]
    logger.remove()
    logger.add(sys.stderr, level=level, format=LOG_FORMAT)
    logger.enable("analyzer")
