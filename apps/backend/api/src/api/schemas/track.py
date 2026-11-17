from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from api.models import (
    TRACK_ARTIST_MAX_LENGTH,
    TRACK_FILENAME_MAX_LENGTH,
    TRACK_NAME_MAX_LENGTH,
    TrackStatus,
)

Title = Annotated[str, Field(min_length=1, max_length=TRACK_NAME_MAX_LENGTH)]
Artist = Annotated[str, Field(max_length=TRACK_ARTIST_MAX_LENGTH)]


class CreateTrackRequest(BaseModel):
    filename: Annotated[
        str, Field(min_length=1, max_length=TRACK_FILENAME_MAX_LENGTH)
    ]
    size: Annotated[int, Field(gt=0)]
    """Size of the audio in bytes."""
    title: Title | None = None


class CreateTrackResponse(BaseModel):
    track_id: UUID
    upload_url: str
    expires_at: datetime


class UploadCompleteResponse(BaseModel):
    track_id: UUID
    status: TrackStatus


class TrackResponse(BaseModel):
    id: UUID
    title: str
    artist: str | None
    original_filename: str
    status: TrackStatus
    rejection_reason: str | None
    created_at: datetime
    updated_at: datetime
    uploaded_at: datetime | None
    validated_at: datetime | None
    removed_at: datetime | None


class TrackListResponse(BaseModel):
    items: list[TrackResponse]
    next_cursor: str | None


class UpdateTrackRequest(BaseModel):
    title: Title | None = None
    artist: Artist | None = None


class PlaybackUrlResponse(BaseModel):
    url: str
    expires_at: datetime
