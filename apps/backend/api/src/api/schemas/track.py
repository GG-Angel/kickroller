from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from api.models import (
    TRACK_ARTIST_MAX_LENGTH,
    TRACK_TITLE_MAX_LENGTH,
    TrackStatus,
)

Title = Annotated[str, Field(min_length=1, max_length=TRACK_TITLE_MAX_LENGTH)]
Artist = Annotated[str, Field(max_length=TRACK_ARTIST_MAX_LENGTH)]


class TrackResponse(BaseModel):
    id: UUID
    title: str
    artist: str | None
    duration_ms: int
    status: TrackStatus
    created_at: datetime
    updated_at: datetime
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
