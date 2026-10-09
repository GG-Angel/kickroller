from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

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
                kind="kick",
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


def test_load_sample_bank_raises_when_path_does_not_match(
    tmp_path: Path,
) -> None:
    config = bank.SampleBankConfig(
        workspace=tmp_path,
        samples=[bank.SampleConfig(kind="kick", paths=[Path("missing.wav")])],
    )

    with pytest.raises(FileNotFoundError, match="No audio files matched"):
        bank.load_sample_bank(config)


def test_load_sample_bank_from_file_uses_cache_until_config_changes(
    tmp_path: Path,
) -> None:
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
        first_bank = bank.load_sample_bank_from_file(bank_path)
        cached_bank = bank.load_sample_bank_from_file(bank_path)
        bank_path.write_text(bank_path.read_text() + "\n")
        refreshed_bank = bank.load_sample_bank_from_file(bank_path)

    assert load_audio.call_count == 2
    assert np.array_equal(
        cached_bank.samples[0].signal, first_bank.samples[0].signal
    )
    assert np.array_equal(
        refreshed_bank.samples[0].signal, first_bank.samples[0].signal
    )


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
