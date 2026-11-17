from datetime import timedelta

from fastapi import Depends
from obstore import sign_async
from obstore.store import S3Store

from api.core.state import AppState, get_app_state


class ObjectStorage:
    """Track audio in the object store."""

    def __init__(
        self, store: S3Store, signer: S3Store, upload_url_ttl: timedelta
    ) -> None:
        self._store = store
        self._signer = signer
        self.upload_url_ttl = upload_url_ttl

    async def sign_put(self, key: str, expires_in: timedelta) -> str:
        """Sign a URL the client can PUT the object to."""
        return await sign_async(self._signer, "PUT", key, expires_in)


def get_object_storage(
    state: AppState = Depends(get_app_state),
) -> ObjectStorage:
    """FastAPI dependency that returns the shared object storage."""
    return state.storage
