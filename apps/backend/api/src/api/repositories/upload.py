from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.db import get_session
from api.models import Upload


class UploadRepository:
    """Database operations for uploads."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, upload: Upload) -> None:
        """Insert an upload and commit."""
        self._session.add(upload)
        await self._session.commit()


def get_upload_repository(
    session: AsyncSession = Depends(get_session),
) -> UploadRepository:
    """FastAPI dependency that returns an upload repository for one request."""
    return UploadRepository(session)
