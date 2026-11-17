from fastapi import Depends
from obstore.store import S3Store

from api.core.state import AppState, get_app_state


class ObjectStorage:
    """Track audio in the object store."""

    def __init__(self, store: S3Store) -> None:
        self._store = store


def get_object_storage(
    state: AppState = Depends(get_app_state),
) -> ObjectStorage:
    """FastAPI dependency that returns the shared object storage."""
    return state.storage
