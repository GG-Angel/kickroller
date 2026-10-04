import re
import tomllib
from pathlib import Path

from services.storage.io import load_audio_file

from .models import Bank, BankConfig, Sample, SampleConfig, SampleKind

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}


def load_samples(config: SampleConfig, kind: SampleKind, root: Path) -> list[Sample]:
    samples: list[Sample] = []
    exclude = re.compile(config.exclude, re.IGNORECASE) if config.exclude else None
    for pattern in config.files:
        for path in root.glob(pattern.as_posix()):
            if (
                not path.is_file()
                or path.suffix.lower() not in AUDIO_SUFFIXES
                or (exclude and exclude.search(path.name))
            ):
                continue  # skip non-audio or excluded files

            audio = load_audio_file(path)
            samples.append(Sample(kind=kind, path=path, bpm=config.bpm, audio=audio))
    return samples


def load_bank(config: BankConfig) -> Bank:
    samples_by_kind: dict[SampleKind, list[Sample]] = {
        "kick": [],
        "loop": [],
        "clap": [],
        "impact": [],
        "hit": [],
    }
    for group in config.samples:
        samples_by_kind[group.kind].extend(
            load_samples(
                group,
                group.kind,
                config.root,
            )
        )
    return Bank(
        kicks=samples_by_kind["kick"],
        loops=samples_by_kind["loop"],
        claps=samples_by_kind["clap"],
        impacts=samples_by_kind["impact"],
        hits=samples_by_kind["hit"],
    )


def load_bank_from_file(path: Path) -> Bank:
    settings = BankConfig.model_validate(tomllib.loads(path.read_text()))
    return load_bank(settings)
