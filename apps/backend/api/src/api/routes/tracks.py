from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from api.models import TrackStatus
from api.schemas import (
    CreateTrackRequest,
    CreateTrackResponse,
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
    UploadCompleteResponse,
)

router = APIRouter(prefix="/tracks", tags=["tracks"])


def not_implemented() -> HTTPException:
    return HTTPException(status.HTTP_501_NOT_IMPLEMENTED)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_track(body: CreateTrackRequest) -> CreateTrackResponse:
    """Make a pending track and sign a PUT URL for the upload."""
    raise not_implemented()


@router.post(
    "/{track_id}/upload-complete", status_code=status.HTTP_202_ACCEPTED
)
async def complete_upload(track_id: UUID) -> UploadCompleteResponse:
    """Do the HEAD check, set the track uploaded, and queue validation."""
    raise not_implemented()


@router.get("/{track_id}")
async def get_track(track_id: UUID) -> TrackResponse:
    """Get one track. The owner's view, so every status is visible."""
    raise not_implemented()


@router.get("")
async def list_tracks(
    owner_id: UUID | None = None,
    status: TrackStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> TrackListResponse:
    """List tracks, newest first."""
    raise not_implemented()


@router.patch("/{track_id}")
async def update_track(
    track_id: UUID, body: UpdateTrackRequest
) -> TrackResponse:
    """Change the title or artist."""
    raise not_implemented()


@router.delete("/{track_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_track(track_id: UUID) -> None:
    """Set the track removed and delete its objects under tracks/{id}/."""
    raise not_implemented()


@router.post("/{track_id}/playback-url")
async def create_playback_url(track_id: UUID) -> PlaybackUrlResponse:
    """Check access and sign a GET URL for playback."""
    raise not_implemented()
