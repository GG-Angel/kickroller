from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import fsspec
from fsspec.implementations.local import LocalFileSystem


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
