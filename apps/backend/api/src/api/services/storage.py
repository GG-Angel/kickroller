from datetime import timedelta

from fastapi import Depends
from obstore import delete_async, sign_async
from obstore.store import ObjectStore, S3Store

from api.core.state import AppState, get_app_state


class StorageService:
    """Track audio in the object store."""

    def __init__(
        self,
        store: ObjectStore,
        signer: S3Store,
        upload_url_ttl: timedelta,
        playback_url_ttl: timedelta,
    ) -> None:
        self._store = store
        self._signer = signer
        self.upload_url_ttl = upload_url_ttl
        self.playback_url_ttl = playback_url_ttl

    async def sign_put(self, key: str, expires_in: timedelta) -> str:
        """Sign a URL the client can PUT the object to."""
        return await sign_async(self._signer, "PUT", key, expires_in)

    async def sign_get(self, key: str, expires_in: timedelta) -> str:
        """Sign a URL the client can GET the object from."""
        return await sign_async(self._signer, "GET", key, expires_in)

    async def delete(self, key: str) -> None:
        """Delete an object. A missing object is not an error on S3."""
        await delete_async(self._store, key)


def get_storage_service(
    state: AppState = Depends(get_app_state),
) -> StorageService:
    """FastAPI dependency that returns the object storage."""
    config = state.config.storage
    return StorageService(
        state.store,
        state.signer,
        config.upload_url_ttl,
        config.playback_url_ttl,
    )
