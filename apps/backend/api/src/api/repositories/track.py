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

    async def list_ready_by_owner(
        self,
        owner_id: UUID,
        limit: int,
        before_track_id: UUID | None,
    ) -> Sequence[Track]:
        """The owner's ready tracks newest first, older than `before_track_id`."""
        # uuidv7 ids are time-ordered, so id order is creation order.
        query = (
            select(Track)
            .where(Track.user_id == owner_id, Track.status == TrackStatus.READY)
            .order_by(Track.id.desc())
            .limit(limit)
        )
        if before_track_id is not None:
            query = query.where(Track.id < before_track_id)
        return (await self._session.scalars(query)).all()

    async def get(self, track_id: UUID) -> Track | None:
        """Load one track, or None if it doesn't exist."""
        return await self._session.get(Track, track_id)

    async def commit(self) -> None:
        """Commit the session, including changes to loaded tracks."""
        await self._session.commit()


def get_track_repository(
    session: AsyncSession = Depends(get_session),
) -> TrackRepository:
    """FastAPI dependency that returns a track repository for one request."""
    return TrackRepository(session)
