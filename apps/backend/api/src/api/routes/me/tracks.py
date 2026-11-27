from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.core.auth import get_user_id
from api.mappers.track import to_track_response
from api.schemas import (
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
)
from api.services.track import TrackService, get_track_service
from kickroller.models import Track

router = APIRouter(prefix="/tracks", tags=["tracks"])


async def get_user_track(
    track_id: UUID,
    user_id: UUID = Depends(get_user_id),
    tracks: TrackService = Depends(get_track_service),
) -> Track:
    """The caller's visible track from the path, or 404."""
    track = await tracks.get_owned_ready_track(user_id, track_id)
    if track is None:  # Same 404 for others' tracks, so ids don't leak.
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return track


@router.get("/{track_id}")
async def get_track(
    track: Track = Depends(get_user_track),
) -> TrackResponse:
    """Get one of your own tracks."""
    return to_track_response(track)


@router.get("")
async def list_tracks(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: UUID | None = None,
    user_id: UUID = Depends(get_user_id),
    tracks: TrackService = Depends(get_track_service),
) -> TrackListResponse:
    """List your own visible tracks, newest first."""
    return await tracks.list_ready_by_owner(user_id, limit, cursor)


@router.patch("/{track_id}")
async def update_track(
    body: UpdateTrackRequest,
    track: Track = Depends(get_user_track),
    tracks: TrackService = Depends(get_track_service),
) -> TrackResponse:
    """Change the title or artist of your own track."""
    return await tracks.update(track, body)


@router.delete("/{track_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_track(
    track: Track = Depends(get_user_track),
    tracks: TrackService = Depends(get_track_service),
) -> None:
    """Remove your own track and delete its audio."""
    await tracks.remove(track)


@router.post("/{track_id}/playback-url")
async def create_playback_url(
    track: Track = Depends(get_user_track),
    tracks: TrackService = Depends(get_track_service),
) -> PlaybackUrlResponse:
    """Sign a GET URL to stream your own track's audio."""
    return await tracks.create_playback_url(track)
