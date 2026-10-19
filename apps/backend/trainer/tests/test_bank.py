from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.models.audio import SampleKind
from src.services.synthesis import bank


def test_load_sample_bank_resolves_workspace_paths_and_globs(
    tmp_path: Path,
) -> None:
    direct_path = tmp_path / "direct.wav"
    globbed_path = tmp_path / "loops" / "loop.wav"
    non_audio_path = tmp_path / "loops" / "notes.txt"
    direct_path.touch()
    globbed_path.parent.mkdir()
    globbed_path.touch()
    non_audio_path.touch()
    config = bank.SampleBankConfig(
        workspace=tmp_path,
        samples=[
            bank.SampleConfig(
                kind=bank.SampleKind.KICK,
                paths=[Path("direct.wav"), Path("loops/*")],
            )
        ],
    )

    with patch(
        "src.services.synthesis.bank.load_audio", return_value=np.zeros(0)
    ) as load_audio:
        sample_bank = bank.load_sample_bank(config)

    assert [call.args[0] for call in load_audio.call_args_list] == [
        direct_path,
        globbed_path,
    ]
    assert [sample.path for sample in sample_bank.samples] == [
        direct_path,
        globbed_path,
    ]
    assert all(
        sample.kind is bank.SampleKind.KICK for sample in sample_bank.samples
    )


def test_load_sample_bank_raises_when_path_does_not_match(
    tmp_path: Path,
) -> None:
    config = bank.SampleBankConfig(
        workspace=tmp_path,
        samples=[
            bank.SampleConfig(
                kind=bank.SampleKind.KICK, paths=[Path("missing.wav")]
            )
        ],
    )

    with pytest.raises(FileNotFoundError, match="No audio files matched"):
        bank.load_sample_bank(config)


def _write_bank_config(
    bank_path: Path, workspace: Path, bpm: int = 160
) -> None:
    bank_path.write_text(
        f"workspace: {workspace}\n"
        "samples:\n"
        "  - kind: kick\n"
        "    paths: [kicks/*]\n"
        f"    bpm: {bpm}\n"
    )


def _decode(path: Path) -> np.ndarray:
    return np.full(4, len(path.name), dtype=np.float32)


def test_load_sample_bank_from_file_reuses_cache(tmp_path: Path) -> None:
    (tmp_path / "kicks").mkdir()
    (tmp_path / "kicks" / "kick.wav").touch()
    bank_path = tmp_path / "bank.yaml"
    _write_bank_config(bank_path, tmp_path)

    with patch(
        "src.services.synthesis.bank.load_audio", side_effect=_decode
    ) as load_audio:
        first_bank = bank.load_sample_bank_from_file(bank_path)
        cached_bank = bank.load_sample_bank_from_file(bank_path)

    assert load_audio.call_count == 1
    assert np.array_equal(
        cached_bank.samples[0].signal, first_bank.samples[0].signal
    )
    assert cached_bank.samples[0].name == "kicks/kick.wav"


def test_load_sample_bank_from_file_refreshes_cache_for_new_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "kicks").mkdir()
    (tmp_path / "kicks" / "kick.wav").touch()
    bank_path = tmp_path / "bank.yaml"
    _write_bank_config(bank_path, tmp_path)

    with patch("src.services.synthesis.bank.load_audio", side_effect=_decode):
        bank.load_sample_bank_from_file(bank_path)
        (tmp_path / "kicks" / "new_kick.wav").touch()
        refreshed_bank = bank.load_sample_bank_from_file(bank_path)

    assert [sample.name for sample in refreshed_bank.samples] == [
        "kicks/kick.wav",
        "kicks/new_kick.wav",
    ]
    assert np.array_equal(refreshed_bank.samples[1].signal, np.full(4, 12))


def test_load_sample_bank_from_file_refreshes_cache_for_new_settings(
    tmp_path: Path,
) -> None:
    (tmp_path / "kicks").mkdir()
    (tmp_path / "kicks" / "kick.wav").touch()
    bank_path = tmp_path / "bank.yaml"
    _write_bank_config(bank_path, tmp_path)

    with patch(
        "src.services.synthesis.bank.load_audio", side_effect=_decode
    ) as load_audio:
        bank.load_sample_bank_from_file(bank_path)
        _write_bank_config(bank_path, tmp_path, bpm=150)
        refreshed_bank = bank.load_sample_bank_from_file(bank_path)

    assert load_audio.call_count == 2
    assert refreshed_bank.samples[0].bpm == 150


def test_load_sample_bank_from_file_replaces_old_cache_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "kicks").mkdir()
    (tmp_path / "kicks" / "kick.wav").touch()
    bank_path = tmp_path / "bank.yaml"
    _write_bank_config(bank_path, tmp_path)
    (tmp_path / "bank.yaml.cache").write_bytes(b"old pickle")

    with patch("src.services.synthesis.bank.load_audio", side_effect=_decode):
        sample_bank = bank.load_sample_bank_from_file(bank_path)

    assert (tmp_path / "bank.yaml.cache").is_dir()
    assert len(sample_bank.samples) == 1


def test_load_sample_bank_from_file_can_bypass_cache(tmp_path: Path) -> None:
    audio_path = tmp_path / "kick.wav"
    audio_path.touch()
    bank_path = tmp_path / "bank.yaml"
    bank_path.write_text(
        f"workspace: {tmp_path}\n"
        "samples:\n"
        "  - kind: kick\n"
        "    paths: [kick.wav]\n"
    )

    with patch(
        "src.services.synthesis.bank.load_audio",
        return_value=np.zeros(4, dtype=np.float32),
    ) as load_audio:
        bank.load_sample_bank_from_file(bank_path)
        bank.load_sample_bank_from_file(bank_path, use_cache=False)

    assert load_audio.call_count == 2


def test_is_held_out_is_fixed_and_holds_out_the_fraction() -> None:
    names = [f"pack/kick_{index}.wav" for index in range(5000)]

    held_out = [bank.is_held_out(name, 0.1) for name in names]

    assert held_out == [bank.is_held_out(name, 0.1) for name in names]
    assert 0.08 < np.mean(held_out) < 0.12
    assert not any(bank.is_held_out(name, 0.0) for name in names)
    assert all(bank.is_held_out(name, 1.0) for name in names)


def _sample(name: str, kind: SampleKind) -> bank.Sample:
    return bank.Sample(
        name=name,
        path=Path(name),
        signal=np.zeros(10, dtype=np.float32),
        kind=kind,
        bpm=160,
        is_loop=False,
    )


def test_select_kicks_splits_only_the_kicks() -> None:
    kicks = [
        _sample(f"kick_{index}.wav", SampleKind.KICK) for index in range(200)
    ]
    others = [
        _sample("clap.wav", SampleKind.DRUM),
        _sample("fx.wav", SampleKind.FX),
    ]
    sample_bank = bank.SampleBank(samples=kicks + others)

    training = sample_bank.select_kicks(held_out=False, fraction=0.2)
    validation = sample_bank.select_kicks(held_out=True, fraction=0.2)

    def kick_names(selected: bank.SampleBank) -> set[str]:
        return {s.name for s in selected.samples if s.kind == SampleKind.KICK}

    assert kick_names(training).isdisjoint(kick_names(validation))
    assert kick_names(training) | kick_names(validation) == kick_names(
        sample_bank
    )
    assert kick_names(validation)
    for selected in (training, validation):
        assert [
            s for s in selected.samples if s.kind != SampleKind.KICK
        ] == others
