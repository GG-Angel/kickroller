from pathlib import Path

import typer

from src.core.config import CONFIG, SynthConfig
from src.core.logging import configure_logging
from src.core.storage import load_audio, load_onsets, save_audio, save_onsets
from src.models.audio import SampleKind
from src.services.analysis.analyze import (
    add_clicks,
    analyze_signal,
)
from src.services.scoring import score_onsets
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
        ..., exists=True, dir_okay=False, help="Audio file."
    ),
    model: Path = typer.Option(
        Path("model.pt"),
        "--model",
        "-m",
        exists=True,
        dir_okay=False,
        help="Model checkpoint.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        help="Onset CSV (default: <audio-name>-onsets.csv).",
    ),
    confidences: bool = typer.Option(
        True,
        "--confidences/--no-confidences",
        help="Include onset confidence.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        min=0,
        max=1,
        help="Peak threshold (default: model setting).",
    ),
    clicks: bool = typer.Option(
        False,
        "--clicks",
        help="Write click audio as <output>.clicks.wav.",
    ),
    labels: Path | None = typer.Option(
        None,
        "--labels",
        exists=True,
        dir_okay=False,
        help="Label CSV for scoring labeled spans.",
    ),
) -> None:
    """Find kick onsets in a track."""
    output = output or track.with_name(f"{track.stem}-onsets.csv")
    checkpoint = load_checkpoint(model)
    signal = load_audio(track, CONFIG.sr)
    analysis = analyze_signal(
        signal, checkpoint, threshold, CONFIG.train, CONFIG.sr
    )
    save_onsets(
        output,
        analysis.onsets,
        analysis.confidences if confidences else None,
    )
    typer.echo(f"Found {len(analysis.onsets)} kicks, saved to {output}.")
    if clicks:
        clicks_path = output.with_suffix(".clicks.wav")
        save_audio(
            clicks_path,
            add_clicks(signal, analysis.onsets, CONFIG.sr),
            CONFIG.sr,
        )
        typer.echo(f"Saved clicks to {clicks_path}.")
    if labels:
        score = score_onsets(
            analysis.onsets,
            load_onsets(labels),
            checkpoint.threshold if threshold is None else threshold,
            CONFIG.train.tolerance,
        )
        typer.echo(
            f"F-measure {score.f_measure:.3f} (precision "
            f"{score.precision:.3f}, recall {score.recall:.3f})."
        )


@app.command()
def train(
    bank_file: Path = typer.Option(
        Path("bank.yaml"),
        "--bank",
        "-b",
        help="Sample bank config.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass the sample cache.",
    ),
    output: Path = typer.Option(
        Path("model.pt"),
        "--output",
        "-o",
        dir_okay=False,
        help="Best checkpoint output.",
    ),
    steps: int = typer.Option(
        CONFIG.train.steps,
        "--steps",
        min=1,
        help="Training steps.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        min=0,
        help="Random seed (default: generated and logged).",
    ),
) -> None:
    """Train a kick-onset model on synthetic drops."""
    config = CONFIG.train.model_copy(update={"steps": steps})
    best = train_model(
        bank_file,
        output,
        not no_cache,
        seed,
        config=config,
        synth=CONFIG.synth,
        feature_config=CONFIG.features,
        sr=CONFIG.sr,
    )
    typer.echo(
        f"Best validation F-measure {best.f_measure:.3f} at threshold "
        f"{best.threshold}, saved to {output}."
    )


@app.command("synth")
def synthesize(
    bank_file: Path = typer.Option(
        Path("bank.yaml"),
        "--bank",
        "-b",
        help="Sample bank config.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Bypass the sample cache.",
    ),
    output: Path = typer.Option(
        Path("drops"),
        "--output",
        "-o",
        file_okay=False,
        help="Drop output folder.",
    ),
    count: int = typer.Option(4, "--count", "-n", min=1, help="Drop count."),
    seed: int | None = typer.Option(
        None,
        "--seed",
        min=0,
        help="Random seed (default: generated and logged).",
    ),
    kinds: list[SampleKind] | None = typer.Option(
        None,
        "--kind",
        "-k",
        help="Layer kind (repeatable; default: all). Kicks always included.",
    ),
    held_out: bool = typer.Option(
        False,
        "--held-out",
        help="Use validation kicks instead of training kicks.",
    ),
    no_master: bool = typer.Option(
        False,
        "--no-master",
        help="Disable master effects.",
    ),
) -> None:
    """Write synthetic drops and their kick onsets."""
    bank = load_sample_bank_from_file(
        bank_file, CONFIG.sr, use_cache=not no_cache
    )
    typer.echo(f"Loaded bank with {len(bank.samples)} samples.")
    bank = bank.select_kicks(held_out, CONFIG.synth.held_out_kick_fraction)
    kicks = sum(sample.kind == SampleKind.KICK for sample in bank.samples)
    typer.echo(f"Using {kicks} {'held-out' if held_out else 'training'} kicks.")
    config = _select_kinds(kinds, CONFIG.synth)
    if no_master:
        config = config.model_copy(update={"master": None})
    export_drops(bank, output, count, seed, config=config, sr=CONFIG.sr)
    typer.echo(f"Wrote {count} drops to {output}.")


def _select_kinds(
    kinds: list[SampleKind] | None, config: SynthConfig
) -> SynthConfig:
    """Keep selected layers, or all layers if none are selected."""
    if kinds is None:
        return config
    layers = [layer for layer in config.layers if layer.kind in kinds]
    return config.model_copy(update={"layers": layers})
