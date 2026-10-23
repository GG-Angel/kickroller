from pathlib import Path

import typer

from src.core.config import CONFIG, SynthConfig
from src.core.logging import configure_logging
from src.core.storage import load_audio, load_onsets, save_audio, save_onsets
from src.models.audio import SampleKind
from src.services.analysis.analyze import (
    add_clicks,
    analyze_signal,
    score_onsets,
)
from src.services.synthesis.bank import load_sample_bank_from_file
from src.services.synthesis.export import export_drops
from src.services.training.network import load_checkpoint
from src.services.training.train import train as train_model

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
def analyze(
    track: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Audio file of the track."
    ),
    model: Path = typer.Option(
        Path("model.pt"),
        "--model",
        "-m",
        exists=True,
        dir_okay=False,
        help="Trained model file.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        help="CSV of `time,confidence` lines (default: the track path, .csv).",
    ),
    confidences: bool = typer.Option(
        True,
        "--confidences/--no-confidences",
        help="Write the confidence of each onset, or only the onset times.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        min=0,
        max=1,
        help="Peak threshold (default: the threshold of the model).",
    ),
    clicks: bool = typer.Option(
        False,
        "--clicks",
        help="Also write the track with a click at each onset, as "
        "`<output>.clicks.wav`.",
    ),
    labels: Path | None = typer.Option(
        None,
        "--labels",
        exists=True,
        dir_okay=False,
        help="CSV of labeled onset times to score the onsets against.",
    ),
) -> None:
    """Find the kick onsets of a track."""
    output = output or track.with_suffix(".csv")
    checkpoint = load_checkpoint(model)
    signal = load_audio(track)
    analysis = analyze_signal(signal, checkpoint, threshold)
    save_onsets(
        output,
        analysis.onsets,
        analysis.confidences if confidences else None,
    )
    typer.echo(f"Found {len(analysis.onsets)} kicks, saved to {output}.")
    if clicks:
        clicks_path = output.with_suffix(".clicks.wav")
        save_audio(clicks_path, add_clicks(signal, analysis.onsets))
        typer.echo(f"Saved clicks to {clicks_path}.")
    if labels:
        score = score_onsets(
            analysis.onsets,
            load_onsets(labels),
            threshold or checkpoint.threshold,
        )
        typer.echo(
            f"F-measure {score.f_measure:.3f} (precision "
            f"{score.precision:.3f}, recall {score.recall:.3f})."
        )


@app.command()
def train(
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
        Path("model.pt"),
        "--output",
        "-o",
        dir_okay=False,
        help="File for the model with the best validation F-measure.",
    ),
    steps: int = typer.Option(
        CONFIG.train.steps, "--steps", min=1, help="Number of training steps."
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        min=0,
        help="Seed for reproducible training (default: a new seed in the log).",
    ),
) -> None:
    """Train a kick onset model on synthetic drops."""
    config = CONFIG.train.model_copy(update={"steps": steps})
    best = train_model(bank_file, output, not no_cache, seed, config)
    typer.echo(
        f"Best validation F-measure {best.f_measure:.3f} at threshold "
        f"{best.threshold}, saved to {output}."
    )


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
        "The kick is always used, unless --kickless, so `-k kick` gives kicks "
        "only.",
    ),
    kickless: bool = typer.Option(
        False,
        "--kickless",
        help="Make every drop kickless: no kick and no onsets, and no "
        "high-pass or sidechain on the layers.",
    ),
    held_out: bool = typer.Option(
        False,
        "--held-out",
        help="Use only the held-out kicks, for validation drops. Without it, "
        "the held-out kicks are never used.",
    ),
    no_master: bool = typer.Option(
        False,
        "--no-master",
        help="Do not augment the drops with random master effects.",
    ),
) -> None:
    """Write synthetic drops with labeled kick onsets."""
    bank = load_sample_bank_from_file(bank_file, use_cache=not no_cache)
    typer.echo(f"Loaded bank with {len(bank.samples)} samples.")
    bank = bank.select_kicks(held_out, CONFIG.synth.held_out_kick_fraction)
    kicks = sum(sample.kind == SampleKind.KICK for sample in bank.samples)
    typer.echo(f"Using {kicks} {'held-out' if held_out else 'training'} kicks.")
    config = _select_kinds(kinds).model_copy(
        update={"kickless_probability": float(kickless)}
    )
    if no_master:
        config = config.model_copy(update={"master": None})
    export_drops(bank, output, count, seed, config)
    typer.echo(f"Wrote {count} drops to {output}.")


def _select_kinds(kinds: list[SampleKind] | None) -> SynthConfig:
    """The synth config with only the layers of ``kinds``, or all layers."""
    if kinds is None:
        return CONFIG.synth
    layers = [layer for layer in CONFIG.synth.layers if layer.kind in kinds]
    return CONFIG.synth.model_copy(update={"layers": layers})
