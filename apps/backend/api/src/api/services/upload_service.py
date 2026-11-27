from datetime import UTC, datetime
from uuid import UUID, uuid7

from api.db.upload_repository import UploadDbRepository
from api.schemas import CreateUploadResponse
from api.services.storage_service import StorageService
from kickroller.models import Upload

# The bucket's expiry and ingest-notification rules match on this prefix.
UPLOAD_OBJECT_KEY_PREFIX = "uploads/"


class UploadService:
    """Upload attempts: their rows in the DB and their objects in storage."""

    def __init__(
        self, uploads: UploadDbRepository, storage: StorageService
    ) -> None:
        self._uploads = uploads
        self._storage = storage

    async def create(
        self, user_id: UUID, filename: str
    ) -> CreateUploadResponse:
        """Insert a pending upload, then sign a POST form for its object."""
        upload_id = uuid7()
        upload = Upload(
            id=upload_id,
            user_id=user_id,
            object_key=f"{UPLOAD_OBJECT_KEY_PREFIX}{upload_id}",
            filename=filename,
            expires_at=datetime.now(UTC) + self._storage.upload_url_ttl,
        )
        await self._uploads.add(upload)

        form = await self._storage.create_post_form(
            key=upload.object_key,
            expires_in=self._storage.upload_url_ttl,
        )

        return CreateUploadResponse(
            upload_id=upload.id,
            upload_url=form.url,
            upload_fields=form.fields,
            expires_at=upload.expires_at,
        )
