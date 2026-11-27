from dataclasses import dataclass
from datetime import timedelta

from fastapi import Depends
from types_aiobotocore_s3 import S3Client

from api.core.state import AppState, get_app_state
from kickroller.models import UPLOAD_MAX_SIZE_BYTES


@dataclass(frozen=True, slots=True)
class PostForm:
    """A signed form the client POSTs a file to."""

    url: str
    fields: dict[str, str]
    """Signed fields to send before the file."""


class StorageService:
    """Track audio in the object store."""

    def __init__(
        self,
        store: S3Client,
        url_signing_store: S3Client,
        bucket: str,
        upload_url_ttl: timedelta,
        playback_url_ttl: timedelta,
    ) -> None:
        self._store = store
        self._url_signing_store = url_signing_store
        self._bucket = bucket
        self.upload_url_ttl = upload_url_ttl
        self.playback_url_ttl = playback_url_ttl

    async def create_post_form(
        self, key: str, expires_in: timedelta
    ) -> PostForm:
        """Sign a form the client POSTs audio to.

        The policy caps the size and requires an `audio/*` Content-Type.
        """
        form = await self._url_signing_store.generate_presigned_post(
            Bucket=self._bucket,
            Key=key,
            Conditions=[
                ["content-length-range", 1, UPLOAD_MAX_SIZE_BYTES],
                ["starts-with", "$Content-Type", "audio/"],
            ],
            ExpiresIn=int(expires_in.total_seconds()),
        )
        return PostForm(url=form["url"], fields=form["fields"])

    async def create_get_url(self, key: str, expires_in: timedelta) -> str:
        """Sign a URL the client can GET the object from."""
        return await self._url_signing_store.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=int(expires_in.total_seconds()),
        )

    async def delete(self, key: str) -> None:
        """Delete an object. A missing object is not an error on S3."""
        await self._store.delete_object(Bucket=self._bucket, Key=key)


def get_storage_service(
    state: AppState = Depends(get_app_state),
) -> StorageService:
    """FastAPI dependency that returns the storage service."""
    storage_config = state.config.storage
    return StorageService(
        state.store,
        state.url_signing_store,
        storage_config.bucket,
        storage_config.upload_url_ttl,
        storage_config.playback_url_ttl,
    )
