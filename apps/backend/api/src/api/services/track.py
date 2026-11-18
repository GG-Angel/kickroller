from uuid import UUID

from fastapi import Depends

from api.models import TrackStatus
from api.repositories.track import TrackRepository, get_track_repository
from api.schemas import TrackListResponse, TrackResponse


class TrackService:
    """Validated tracks."""

    def __init__(self, tracks: TrackRepository) -> None:
        self._tracks = tracks

    async def list(
        self,
        owner_id: UUID | None,
        status: TrackStatus | None,
        limit: int,
        cursor: UUID | None,
    ) -> TrackListResponse:
        """One page of tracks, newest first. The cursor is the last id seen."""
        # One extra row tells us whether another page exists.
        rows = await self._tracks.list(owner_id, status, limit + 1, cursor)
        page = rows[:limit]
        return TrackListResponse(
            items=[
                TrackResponse.model_validate(t, from_attributes=True)
                for t in page
            ],
            next_cursor=str(page[-1].id) if len(rows) > limit else None,
        )


def get_track_service(
    tracks: TrackRepository = Depends(get_track_repository),
) -> TrackService:
    """FastAPI dependency that returns a track service for one request."""
    return TrackService(tracks)
