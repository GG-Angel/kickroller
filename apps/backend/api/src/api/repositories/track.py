from collections.abc import Sequence
from uuid import UUID

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.db import get_session
from api.models import Track, TrackStatus


class TrackRepository:
    """Database operations for tracks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(
        self,
        owner_id: UUID,
        limit: int,
        before: UUID | None,
    ) -> Sequence[Track]:
        """The owner's ready tracks newest first, older than `before` if set."""
        # uuidv7 ids are time-ordered, so id order is creation order.
        query = (
            select(Track)
            .where(Track.user_id == owner_id, Track.status == TrackStatus.READY)
            .order_by(Track.id.desc())
            .limit(limit)
        )
        if before is not None:
            query = query.where(Track.id < before)
        return (await self._session.scalars(query)).all()

    async def get(self, track_id: UUID) -> Track | None:
        """Load one track, or None if it doesn't exist."""
        return await self._session.get(Track, track_id)

    async def save(self) -> None:
        """Commit changes made to loaded tracks."""
        await self._session.commit()


def get_track_repository(
    session: AsyncSession = Depends(get_session),
) -> TrackRepository:
    """FastAPI dependency that returns a track repository for one request."""
    return TrackRepository(session)
