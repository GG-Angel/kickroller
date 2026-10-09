import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.core.storage import load_audio, load_yaml
from src.models.audio import Signal

AUDIO_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".aifc",
    ".au",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
}

SampleKind = Literal[
    "kick",
    "melody",
    "vocal",
    "drum",
    "bass",
    "impact",
    "fx",
]


@dataclass(frozen=True)
class Sample:
    name: str
    path: Path
    signal: Signal
    kind: SampleKind

    bpm: int
    is_loop: bool

    @property
    def is_one_shot(self) -> bool:
        return not self.is_loop


@dataclass(frozen=True)
class SampleBank:
    samples: list[Sample]

    @property
    def kicks(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.kind == "kick"]

    @property
    def loops(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.is_loop]

    @property
    def one_shots(self) -> list[Sample]:
        return [sample for sample in self.samples if sample.is_one_shot]


class SampleConfig(BaseModel):
    kind: SampleKind
    paths: list[Path]

    bpm: int = 160
    is_loop: bool = False


class SampleBankConfig(BaseModel):
    workspace: Path | None = None
    samples: list[SampleConfig]


def load_sample_bank(config: SampleBankConfig) -> SampleBank:
    samples: list[Sample] = []
    for sample_config in config.samples:
        for configured_path in sample_config.paths:
            path_pattern = (
                config.workspace / configured_path
                if config.workspace
                else configured_path
            )
            paths = sorted(
                Path(path)
                for path in glob.glob(str(path_pattern))
                if Path(path).suffix.lower() in AUDIO_EXTENSIONS
            )
            if not paths:
                raise FileNotFoundError(
                    f"No audio files matched sample path: {path_pattern}"
                )

            for path in paths:
                name = path.stem
                signal = load_audio(path)
                sample = Sample(
                    name=name,
                    path=path,
                    signal=signal,
                    kind=sample_config.kind,
                    bpm=sample_config.bpm,
                    is_loop=sample_config.is_loop,
                )
                samples.append(sample)
    return SampleBank(samples=samples)


def load_sample_bank_config(path: Path) -> SampleBankConfig:
    return SampleBankConfig.model_validate(load_yaml(path))


def load_sample_bank_from_file(path: Path) -> SampleBank:
    config = load_sample_bank_config(path)
    return load_sample_bank(config)
