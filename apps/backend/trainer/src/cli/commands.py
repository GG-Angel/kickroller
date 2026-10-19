from pathlib import Path

import typer

from src.core.config import CONFIG, SynthConfig
from src.core.logging import configure_logging
from src.models.audio import SampleKind
from src.services.synthesis.bank import load_sample_bank_from_file
from src.services.synthesis.export import export_drops

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
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Decode samples without reading or writing the bank cache.",
    ),
    output: Path = typer.Option(
        Path("drops"),
        "--output",
        "-o",
        file_okay=False,
        help="Folder for the drops and their kick onsets.",
    ),
    count: int = typer.Option(
        4, "--count", "-n", min=1, help="Number of drops to write."
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        min=0,
        help="Seed for reproducible drops (default: a new seed in the log).",
    ),
    kinds: list[SampleKind] | None = typer.Option(
        None,
        "--kind",
        "-k",
        help="Only use layers of this kind (repeatable; default: all). "
        "The kick is always used, so `-k kick` gives kicks only.",
    ),
    held_out: bool = typer.Option(
        False,
        "--held-out",
        help="Use only the held-out kicks, for validation drops. Without it, "
        "the held-out kicks are never used.",
    ),
) -> None:
    """Write synthetic drops with labeled kick onsets."""
    bank = load_sample_bank_from_file(bank_file, use_cache=not no_cache)
    typer.echo(f"Loaded bank with {len(bank.samples)} samples.")
    bank = bank.select_kicks(held_out, CONFIG.synth.held_out_kick_fraction)
    kicks = sum(sample.kind == SampleKind.KICK for sample in bank.samples)
    typer.echo(f"Using {kicks} {'held-out' if held_out else 'training'} kicks.")
    export_drops(bank, output, count, seed, _select_kinds(kinds))
    typer.echo(f"Wrote {count} drops to {output}.")


def _select_kinds(kinds: list[SampleKind] | None) -> SynthConfig:
    """The synth config with only the layers of ``kinds``, or all layers."""
    if kinds is None:
        return CONFIG.synth
    layers = [layer for layer in CONFIG.synth.layers if layer.kind in kinds]
    return CONFIG.synth.model_copy(update={"layers": layers})
