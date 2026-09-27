import argparse
import json
import sys
from pathlib import Path

from analyzer.audio import load_mid, normalize_loudness
from analyzer.detect import DetectorConfig, detect_kicks_in_signal
from analyzer.sonify import sonify, write_wav


def format_csv(times, confidence) -> str:
    return "".join(f"{t:.3f},{c:.2f}\n" for t, c in zip(times, confidence))


def format_json(path: Path, times, confidence) -> str:
    kicks = [
        {"time": round(float(t), 3), "confidence": round(float(c), 2)}
        for t, c in zip(times, confidence)
    ]
    return json.dumps({"file": path.name, "kicks": kicks}) + "\n"


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
        help="csv: one 'time,confidence' per line, time in seconds; "
        "json: {file, kicks: [{time, confidence}]} (default: csv)",
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=defaults.min_confidence,
        help="only output kicks with at least this confidence, 0-1 "
        f"(default: {defaults.min_confidence})",
    )
    parser.add_argument(
        "--sonify",
        type=Path,
        metavar="WAV",
        help="also write a WAV file of the track with a click at each detected kick "
        "(louder for higher confidence)",
    )
    args = parser.parse_args(argv)

    config = DetectorConfig(min_confidence=args.min_confidence)
    try:
        signal = load_mid(args.audio, config.sample_rate)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"analyzer: error: {error}", file=sys.stderr)
        return 1
    times, confidence = detect_kicks_in_signal(signal, config)

    text = (
        format_json(args.audio, times, confidence)
        if args.format == "json"
        else format_csv(times, confidence)
    )
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)

    if args.sonify:
        normalized = normalize_loudness(signal, config.sample_rate, config.target_lufs)
        write_wav(
            args.sonify,
            sonify(normalized, times, config.sample_rate, confidence=confidence),
            config.sample_rate,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
