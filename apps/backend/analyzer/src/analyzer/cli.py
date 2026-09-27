import argparse
import json
import sys
from pathlib import Path

from analyzer.audio import load_mid, normalize_loudness
from analyzer.detect import DetectorConfig, detect_kicks_in_signal
from analyzer.sonify import sonify, write_wav


def format_csv(times) -> str:
    return "".join(f"{t:.3f}\n" for t in times)


def format_json(path: Path, times) -> str:
    return (
        json.dumps({"file": path.name, "kicks": [round(float(t), 3) for t in times]})
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    defaults = DetectorConfig()
    parser = argparse.ArgumentParser(
        prog="analyzer", description="Detect kick onsets in a rawstyle track."
    )
    parser.add_argument(
        "audio", type=Path, help="audio or video file (any format ffmpeg reads)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output file (default: standard output)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="csv: one time in seconds per line (Sonic Visualiser time instants); "
        "json: {file, kicks} (default: csv)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=defaults.threshold,
        help=f"peak threshold on the click-band onset strength (default: {defaults.threshold})",
    )
    parser.add_argument(
        "--sonify",
        type=Path,
        metavar="WAV",
        help="also write a WAV file of the track with a click at each detected kick",
    )
    args = parser.parse_args(argv)

    config = DetectorConfig(threshold=args.threshold)
    try:
        signal = load_mid(args.audio, config.sample_rate)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"analyzer: error: {error}", file=sys.stderr)
        return 1
    times = detect_kicks_in_signal(signal, config)

    text = (
        format_json(args.audio, times) if args.format == "json" else format_csv(times)
    )
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)

    if args.sonify:
        normalized = normalize_loudness(signal, config.sample_rate, config.target_lufs)
        write_wav(
            args.sonify,
            sonify(normalized, times, config.sample_rate),
            config.sample_rate,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
