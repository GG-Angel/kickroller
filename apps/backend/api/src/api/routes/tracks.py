from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.models import TrackStatus
from api.schemas import (
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
)
from api.services.storage import ObjectStorage, get_object_storage

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
    cursor: str | None = None,
    storage: ObjectStorage = Depends(get_object_storage),
) -> TrackListResponse:
    """List tracks, newest first."""
    raise not_implemented()


@router.patch("/{track_id}")
async def update_track(
    track_id: UUID,
    body: UpdateTrackRequest,
    storage: ObjectStorage = Depends(get_object_storage),
) -> TrackResponse:
    """Change the title or artist."""
    raise not_implemented()


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
