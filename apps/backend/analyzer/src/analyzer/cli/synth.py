"""The synth command: write synthetic training drops, to check the training data."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from analyzer.audio.io import SAMPLE_RATE, write_wav
from analyzer.audio.sonify import sonify
from analyzer.cli.options import (
    MODELS,
    BankConfigFile,
    Cache,
    Quiet,
    Verbose,
    configure_logging,
)
from analyzer.training.bank import build_bank, read_config


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


def synth(
    config: BankConfigFile,
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
    cache: Cache = MODELS / "bank",
    verbose: Verbose = 0,
    quiet: Quiet = False,
) -> None:
    """Write synthetic training drops with their kick onsets, to check the training data.

    For each drop: NAME.wav, NAME.csv (one kick onset time per line, in seconds),
    NAME.clicks.wav (a click at each kick) and NAME.json (the onsets, the samples
    and the settings).
    """
    from analyzer.training.synth import Catalog, make_drop

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
