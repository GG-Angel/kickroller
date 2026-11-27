from sqlalchemy.ext.asyncio import AsyncSession

from kickroller.models import Upload


class UploadDbRepository:
    """Database operations for uploads."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, upload: Upload) -> None:
        """Insert an upload and commit."""
        self._session.add(upload)
        await self._session.commit()
