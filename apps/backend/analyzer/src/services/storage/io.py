from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import fsspec
import librosa
import numpy as np
from fsspec.implementations.local import LocalFileSystem
from scipy.io import wavfile

from core.settings import SETTINGS


@contextmanager
def get_file(uri: str | Path, **storage_options: object) -> Generator[Path]:
    """A local path to the file at `uri` (a path, file://, s3://, gs://, ...).

    A local file is used where it is. A remote file is downloaded to a
    temporary folder, which is deleted at the end of the `with` block.
    `storage_options` go to the filesystem (for example `endpoint_url` for MinIO).
    """
    fs, path = fsspec.url_to_fs(url=str(uri), **storage_options)
    if isinstance(fs, LocalFileSystem):
        yield Path(path)
        return
    with TemporaryDirectory() as folder:
        local = Path(folder) / PurePosixPath(path).name  # keep the name and suffix
        fs.get_file(path, str(local))
        yield local


def load_audio_file(
    path: Path,
    sample_rate: int = SETTINGS.sample_rate,
) -> np.ndarray:
    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return signal


def save_audio_file(
    path: Path,
    audio: np.ndarray,
    sample_rate: int = SETTINGS.sample_rate,
) -> None:
    wavfile.write(path, sample_rate, audio.astype(np.float32))
