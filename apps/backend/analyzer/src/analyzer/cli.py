import json
import sys
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from loguru import logger

from analyzer.audio import load_mid, normalize_loudness
from analyzer.detect import Detection, DetectorConfig, detect_kicks_in_signal
from analyzer.sonify import sonify, write_wav

DEFAULTS = DetectorConfig()
LOG_LEVELS = ("INFO", "DEBUG", "TRACE")
LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def cli() -> None:
    """Analyze rawstyle tracks."""


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
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Show more logs: -v for the pipeline steps, "
            "-vv also for each candidate attack.",
        ),
    ] = 0,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only show warnings and errors."),
    ] = False,
) -> None:
    """Detect kick onsets in a rawstyle track."""
    configure_logging(verbose, quiet)
    start = perf_counter()
    config = DetectorConfig(min_confidence=min_confidence)
    logger.debug("Detector settings: {config}", config=config)
    try:
        signal = load_mid(audio, config.sample_rate)
    except (FileNotFoundError, RuntimeError) as error:
        logger.error("{error}", error=error)
        raise typer.Exit(code=1) from error
    detection = detect_kicks_in_signal(signal, config)

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


def main() -> None:
    app()
