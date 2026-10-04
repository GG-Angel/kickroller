import re
import tomllib
from pathlib import Path

from services.storage.io import load_audio_file

from .models import Bank, BankSampleSettings, BankSettings, Sample, SampleKind

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}


def get_samples(
    settings: BankSampleSettings, kind: SampleKind, root: Path
) -> list[Sample]:
    samples: list[Sample] = []
    exclude = re.compile(settings.exclude, re.IGNORECASE) if settings.exclude else None
    for pattern in settings.files:
        for path in root.glob(pattern.as_posix()):
            if (
                not path.is_file()
                or path.suffix.lower() not in AUDIO_SUFFIXES
                or (exclude and exclude.search(path.name))
            ):
                continue  # skip non-audio or excluded files

            audio = load_audio_file(path)
            samples.append(Sample(kind=kind, path=path, bpm=settings.bpm, audio=audio))
    return samples


def get_bank(settings: BankSettings) -> Bank:
    samples_by_kind: dict[SampleKind, list[Sample]] = {
        "kick": [],
        "loop": [],
        "clap": [],
        "impact": [],
        "hit": [],
    }
    for group in settings.samples:
        samples_by_kind[group.kind].extend(
            get_samples(
                group,
                group.kind,
                settings.root,
            )
        )
    return Bank(
        kicks=samples_by_kind["kick"],
        loops=samples_by_kind["loop"],
        claps=samples_by_kind["clap"],
        impacts=samples_by_kind["impact"],
        hits=samples_by_kind["hit"],
    )


def read_bank(path: Path) -> Bank:
    settings = BankSettings.model_validate(tomllib.loads(path.read_text()))
    return get_bank(settings)
