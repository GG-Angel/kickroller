from datetime import UTC, datetime
from uuid import UUID, uuid7

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.db import get_session
from api.models import Upload
from api.schemas import CreateUploadResponse
from api.services.storage import ObjectStorage, get_object_storage


class UploadService:
    """Upload attempts: their rows in the DB and their objects in storage."""

    def __init__(self, session: AsyncSession, storage: ObjectStorage) -> None:
        self._session = session
        self._storage = storage

    async def create(
        self, user_id: UUID, filename: str, size: int
    ) -> CreateUploadResponse:
        """Insert a pending upload, then sign a PUT URL for its object."""
        upload_id = uuid7()
        upload = Upload(
            id=upload_id,
            user_id=user_id,
            object_key=f"uploads/{upload_id}",
            size=size,
            original_filename=filename,
            expires_at=datetime.now(UTC) + self._storage.upload_url_ttl,
        )
        self._session.add(upload)
        await self._session.commit()

        url = await self._storage.sign_put(
            key=upload.object_key,
            expires_in=self._storage.upload_url_ttl,
        )

        return CreateUploadResponse(
            upload_id=upload.id,
            upload_url=url,
            expires_at=upload.expires_at,
        )


def get_upload_service(
    session: AsyncSession = Depends(get_session),
    storage: ObjectStorage = Depends(get_object_storage),
) -> UploadService:
    """FastAPI dependency that returns an upload service for one request."""
    return UploadService(session, storage)
