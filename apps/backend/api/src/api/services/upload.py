from datetime import UTC, datetime
from uuid import UUID, uuid7

from fastapi import Depends

from api.repositories.upload import UploadRepository, get_upload_repository
from api.schemas import CreateUploadResponse
from api.services.storage import StorageService, get_storage_service
from kickroller.models import Upload

# The bucket's expiry and ingest-notification rules match on this prefix.
UPLOAD_OBJECT_KEY_PREFIX = "uploads/"


class UploadService:
    """Upload attempts: their rows in the DB and their objects in storage."""

    def __init__(
        self, uploads: UploadRepository, storage: StorageService
    ) -> None:
        self._uploads = uploads
        self._storage = storage

    async def create(
        self, user_id: UUID, filename: str
    ) -> CreateUploadResponse:
        """Insert a pending upload, then sign a PUT URL for its object."""
        upload_id = uuid7()
        upload = Upload(
            id=upload_id,
            user_id=user_id,
            object_key=f"{UPLOAD_OBJECT_KEY_PREFIX}{upload_id}",
            filename=filename,
            expires_at=datetime.now(UTC) + self._storage.upload_url_ttl,
        )
        await self._uploads.add(upload)

        upload_url = await self._storage.create_put_url(
            key=upload.object_key,
            expires_in=self._storage.upload_url_ttl,
        )

        return CreateUploadResponse(
            upload_id=upload.id,
            upload_url=upload_url,
            expires_at=upload.expires_at,
        )


def get_upload_service(
    uploads: UploadRepository = Depends(get_upload_repository),
    storage: StorageService = Depends(get_storage_service),
) -> UploadService:
    """FastAPI dependency that returns an upload service for one request."""
    return UploadService(uploads, storage)
