from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.core.auth import get_user_id
from api.models import TrackStatus
from api.schemas import (
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
)
from api.services.storage import ObjectStorage, get_object_storage
from api.services.track import TrackService, get_track_service

router = APIRouter(prefix="/tracks", tags=["tracks"])


def not_implemented() -> HTTPException:
    return HTTPException(status.HTTP_501_NOT_IMPLEMENTED)


@router.get("/{track_id}")
async def get_track(
    track_id: UUID, storage: ObjectStorage = Depends(get_object_storage)
) -> TrackResponse:
    """Get one track."""
    raise not_implemented()


@router.get("")
async def list_tracks(
    owner_id: UUID | None = None,
    status: TrackStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: UUID | None = None,
    tracks: TrackService = Depends(get_track_service),
) -> TrackListResponse:
    """List tracks, newest first."""
    return await tracks.list(owner_id, status, limit, cursor)


@router.patch("/{track_id}")
async def update_track(
    track_id: UUID,
    body: UpdateTrackRequest,
    user_id: UUID = Depends(get_user_id),
    tracks: TrackService = Depends(get_track_service),
) -> TrackResponse:
    """Change the title or artist of your own track."""
    track = await tracks.update(user_id, track_id, body)
    if track is None:  # Same 404 for others' tracks, so ids don't leak.
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return track


@router.delete("/{track_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_track(
    track_id: UUID, storage: ObjectStorage = Depends(get_object_storage)
) -> None:
    """Set the track removed and delete its objects under tracks/{id}/."""
    raise not_implemented()


@router.post("/{track_id}/playback-url")
async def create_playback_url(
    track_id: UUID, storage: ObjectStorage = Depends(get_object_storage)
) -> PlaybackUrlResponse:
    """Check access and sign a GET URL for playback."""
    raise not_implemented()
