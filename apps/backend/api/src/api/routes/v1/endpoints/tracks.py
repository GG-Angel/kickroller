from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from api.routes.dependencies import (
    get_track_service,
    get_user_id,
    get_user_track,
)
from api.schemas import (
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
    to_track_response,
)
from api.services.track_service import TrackService
from kickroller.models import Track

router = APIRouter(prefix="/tracks", tags=["tracks"])


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
