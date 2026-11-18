from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.core.auth import get_user_id
from api.models import Track, TrackStatus
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


async def get_own_track(
    track_id: UUID,
    user_id: UUID = Depends(get_user_id),
    tracks: TrackService = Depends(get_track_service),
) -> Track:
    """The caller's visible track from the path, or 404."""
    track = await tracks.get_owned(user_id, track_id)
    if track is None:  # Same 404 for others' tracks, so ids don't leak.
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return track


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
    body: UpdateTrackRequest,
    track: Track = Depends(get_own_track),
    tracks: TrackService = Depends(get_track_service),
) -> TrackResponse:
    """Change the title or artist of your own track."""
    return await tracks.update(track, body)


@router.delete("/{track_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_track(
    track: Track = Depends(get_own_track),
    tracks: TrackService = Depends(get_track_service),
) -> None:
    """Remove your own track and delete its audio."""
    await tracks.delete(track)


@router.post("/{track_id}/playback-url")
async def create_playback_url(
    track: Track = Depends(get_own_track),
    tracks: TrackService = Depends(get_track_service),
) -> PlaybackUrlResponse:
    """Sign a GET URL to stream your own track's audio."""
    return await tracks.create_playback_url(track)
