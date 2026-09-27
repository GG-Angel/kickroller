import json
import sys
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from analyzer.audio import SAMPLE_RATE, load_mid, normalize_loudness
from analyzer.bank import build_bank, read_config
from analyzer.detect import Detection, DetectorConfig, detect_kicks_in_signal
from analyzer.model import DEFAULT_MODEL, load_model
from analyzer.sonify import sonify, write_wav

MODELS = DEFAULT_MODEL.parent

DEFAULTS = DetectorConfig()
LOG_LEVELS = ("INFO", "DEBUG", "TRACE")
LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def cli() -> None:
    """Find the kicks and the beat grid of rawstyle tracks."""


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


def configure_logging(verbose: int, quiet: bool) -> None:
    """Log to standard error, so that standard output only has the result."""
    level = "WARNING" if quiet else LOG_LEVELS[min(verbose, len(LOG_LEVELS) - 1)]
    logger.remove()
    logger.add(sys.stderr, level=level, format=LOG_FORMAT)
    logger.enable("analyzer")


class OutputFormat(StrEnum):
    csv = "csv"
    json = "json"


def format_csv(detection: Detection) -> str:
    return "".join(
        f"{t:.3f},{c:.2f}\n" for t, c in zip(detection.kicks, detection.confidence)
    )


def format_beats(detection: Detection) -> str:
    return "".join(f"{t:.3f}\n" for t in detection.beats)


def format_json(path: Path, detection: Detection) -> str:
    beats = [round(float(t), 3) for t in detection.beats]
    kicks = [
        {"time": round(float(t), 3), "confidence": round(float(c), 2)}
        for t, c in zip(detection.kicks, detection.confidence)
    ]
    return json.dumps({"file": path.name, "beats": beats, "kicks": kicks}) + "\n"


@app.command()
def analyze(
    audio: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Audio or video file (any format ffmpeg reads).",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file (default: standard output)."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="csv: one kick 'time,confidence' per line, time in seconds; "
            "json: the file name, the beat times and the kicks (time, confidence).",
        ),
    ] = OutputFormat.csv,
    min_confidence: Annotated[
        float,
        typer.Option(
            "--min-confidence",
            "-c",
            min=0.0,
            max=1.0,
            help="Only output kicks with at least this confidence.",
        ),
    ] = DEFAULTS.min_confidence,
    beats: Annotated[
        Path | None,
        typer.Option(
            "--beats",
            "-b",
            metavar="FILE",
            help="Also write the beat grid times, one time in seconds per line.",
        ),
    ] = None,
    sonify_path: Annotated[
        Path | None,
        typer.Option(
            "--sonify",
            metavar="WAV",
            help="Also write a WAV file of the track with a click at each detected "
            "kick (louder for higher confidence).",
        ),
    ] = None,
    model_path: Annotated[
        Path,
        typer.Option("--model", "-m", help="The trained kick model."),
    ] = DEFAULT_MODEL,
    verbose: Verbose = 0,
    quiet: Quiet = False,
) -> None:
    """Detect kick onsets and the beat grid in a rawstyle track."""
    configure_logging(verbose, quiet)
    start = perf_counter()
    config = DetectorConfig(min_confidence=min_confidence)
    logger.debug("Detector settings: {config}", config=config)
    try:
        model = load_model(model_path)
        signal = load_mid(audio, config.sample_rate)
    except (FileNotFoundError, RuntimeError) as error:
        logger.error("{error}", error=error)
        raise typer.Exit(code=1) from error
    logger.debug("Loaded the kick model from {path}", path=model_path)
    detection = detect_kicks_in_signal(signal, model, config)

    text = (
        format_json(audio, detection)
        if output_format is OutputFormat.json
        else format_csv(detection)
    )
    if output:
        output.write_text(text)
        logger.info(
            "Wrote {count} kicks to {path}", count=len(detection.kicks), path=output
        )
    else:
        typer.echo(text, nl=False)
    if beats:
        beats.write_text(format_beats(detection))
        logger.info(
            "Wrote {count} beats to {path}", count=len(detection.beats), path=beats
        )

    if sonify_path:
        normalized = normalize_loudness(signal, config.sample_rate, config.target_lufs)
        write_wav(
            sonify_path,
            sonify(
                normalized,
                detection.kicks,
                config.sample_rate,
                confidence=detection.confidence,
            ),
            config.sample_rate,
        )
        logger.info("Wrote the click track to {path}", path=sonify_path)

    logger.info("Done in {seconds:.1f} s", seconds=perf_counter() - start)


class Device(StrEnum):
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="The sample bank config (TOML, see bank.example.toml).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to save the model."),
    ] = DEFAULT_MODEL,
    cache: Annotated[
        Path,
        typer.Option(
            "--cache",
            help="Cache folder for the decoded samples (made on the first run).",
        ),
    ] = MODELS / "bank",
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
    from analyzer.train import train as train_model

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


def log_drop(name: str, info: dict) -> None:
    """The kick design at INFO; the loops, one-shots and mastering at DEBUG (-v)."""
    logger.info(
        "Wrote {name}: {kicks} kicks at {bpm:.1f} BPM, kick design {design}",
        name=name,
        kicks=info["kicks"],
        bpm=info["bpm"],
        design=info["kick_design"],
    )
    for path in info["kick_files"]:
        logger.debug("  kick {path}", path=path)
    for loop in info["loops"]:
        logger.debug(
            "  loop {path} ({filter}, {level:+.1f} dB, {ducked})",
            path=loop["path"],
            filter=f"high-passed at {loop['high_pass_hz']} Hz"
            if loop["high_pass_hz"]
            else "not high-passed",
            level=loop["level_db"],
            ducked="ducked" if loop["ducked"] else "not ducked",
        )
    if info["clap"]:
        logger.debug("  clap {path}", path=info["clap"]["path"])
    for hit in info["hits"]:
        logger.debug(
            "  hit at {time:.2f} s: {path}", time=hit["time"], path=hit["path"]
        )
    if info["impact"]:
        logger.debug(
            "  impact at {time:.2f} s: {path}",
            time=info["impact"]["time"],
            path=info["impact"]["path"],
        )
    master = info["master"]
    logger.debug(
        "  master: EQ tilt {tilt}, drive {drive:.1f} dB, clipper {clipper}, "
        "limiter {limiter}, gain {gain:+.1f} dB",
        tilt="on" if master["eq_tilt"] else "off",
        drive=master["drive_db"],
        clipper="on" if master["clipper"] else "off",
        limiter="on" if master["limiter"] else "off",
        gain=info["gain_db"],
    )


@app.command()
def synth(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="The sample bank config (TOML, see bank.example.toml).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", file_okay=False, help="Output folder."),
    ] = MODELS / "drops",
    count: Annotated[
        int, typer.Option("--count", "-n", min=1, help="Number of drops.")
    ] = 4,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    held_out: Annotated[
        bool,
        typer.Option(
            "--held-out",
            help="Use only held-out kick designs. With the training seed, these "
            "are the validation drops.",
        ),
    ] = False,
    cache: Annotated[
        Path,
        typer.Option(
            "--cache",
            help="Cache folder for the decoded samples (made on the first run).",
        ),
    ] = MODELS / "bank",
    verbose: Verbose = 0,
    quiet: Quiet = False,
) -> None:
    """Write synthetic training drops with their kick onsets, to check the training data.

    For each drop: NAME.wav, NAME.csv (one kick onset time per line, in seconds),
    NAME.clicks.wav (a click at each kick) and NAME.json (the onsets, the samples
    and the settings).
    """
    from analyzer.synth import Catalog, make_drop

    configure_logging(verbose, quiet)
    try:
        bank_config = read_config(config)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="CONFIG") from error
    try:
        catalog = Catalog(build_bank(bank_config, cache))
    except (FileNotFoundError, RuntimeError) as error:
        logger.error("{error}", error=error)
        raise typer.Exit(code=1) from error
    if not catalog.designs[held_out]:
        logger.error(
            "The bank has no {kind} kick designs",
            kind="held-out" if held_out else "training",
        )
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        drop = make_drop(rng, catalog, held_out=held_out)
        name = f"drop_{i:03d}"
        write_wav(output / f"{name}.wav", drop.audio, SAMPLE_RATE)
        write_wav(
            output / f"{name}.clicks.wav",
            sonify(drop.audio, drop.onsets, SAMPLE_RATE),
            SAMPLE_RATE,
        )
        (output / f"{name}.csv").write_text("".join(f"{t:.4f}\n" for t in drop.onsets))
        onsets = [round(float(t), 4) for t in drop.onsets]
        (output / f"{name}.json").write_text(
            json.dumps({"onsets": onsets, **drop.info}, indent=2) + "\n"
        )
        log_drop(name, drop.info)
    logger.info("Wrote {count} drops to {path}", count=count, path=output)


def main() -> None:
    app()
