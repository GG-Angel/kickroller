from pathlib import Path

import numpy as np

from src.services.synthesis import synth
from src.services.synthesis.bank import Sample, SampleBank
from src.services.synthesis.synth import Grid, Track


def test_track_insert_mixes_overlapping_audio_as_float32() -> None:
    track = Track(grid=Grid(bpm=60, sr=8), bars=1)
    signal = np.full(8, 0.5, dtype=np.float32)

    track.insert(signal, beat=0)
    track.insert(signal, beat=0.5)

    np.testing.assert_array_equal(track.signal[:4], np.full(4, 0.5))
    np.testing.assert_array_equal(track.signal[4:8], np.full(4, 1.0))
    assert track.signal.dtype == np.float32


def test_generate_drop_scales_overlapping_audio_to_prevent_clipping(
    monkeypatch,
) -> None:
    sample = Sample(
        name="kick",
        path=Path("kick.wav"),
        signal=np.full(2, 0.75, dtype=np.float32),
        kind="kick",
        bpm=60,
        is_loop=False,
    )
    bank = SampleBank(samples=[sample])
    monkeypatch.setattr(synth, "draw_grid", lambda: Grid(bpm=60, sr=8))
    monkeypatch.setattr(
        synth, "generate_kick_pattern", lambda: np.array([0, 0])
    )

    drop = synth.generate_drop(bank)

    assert drop.dtype == np.float32
    assert np.max(np.abs(drop)) == 1
    np.testing.assert_array_equal(drop[:2], np.ones(2, dtype=np.float32))