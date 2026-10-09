from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.services.synthesis import synth
from src.services.synthesis.bank import Sample, SampleBank


class FakeRenderEngine:
    def __init__(self, sample_rate: int, block_size: int) -> None:
        self.audio = np.zeros((2, 0), dtype=np.float32)

    def set_bpm(self, bpm: float) -> None:
        pass

    def make_playback_processor(
        self, name: str, audio: np.ndarray
    ) -> np.ndarray:
        self.audio = audio
        return audio

    def load_graph(self, graph: list[tuple[np.ndarray, list]]) -> None:
        pass

    def render(self, duration: int, *, beats: bool) -> None:
        pass

    def get_audio(self) -> np.ndarray:
        return self.audio


def make_bank(signal: np.ndarray) -> SampleBank:
    return SampleBank(
        samples=[
            Sample(
                name="sample",
                path=Path("sample.wav"),
                signal=signal,
                kind="kick",
                bpm=160,
                is_loop=False,
            )
        ]
    )


def test_synthesize_places_samples_on_each_beat() -> None:
    sample_rate = 8
    signal = np.ones(2, dtype=np.float32)
    engine_module = SimpleNamespace(RenderEngine=FakeRenderEngine)

    with patch.dict("sys.modules", {"dawdreamer": engine_module}):
        drop = synth.DropSynthesizer(sample_rate).synthesize(
            make_bank(signal), bpm=60, bars=1, seed=1
        )

    assert drop.tolist() == [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_synthesize_requires_samples() -> None:
    with pytest.raises(ValueError, match="empty sample bank"):
        synth.DropSynthesizer().synthesize(SampleBank(samples=[]))