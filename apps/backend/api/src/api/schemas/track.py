from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, StringConstraints, field_validator

from api.models import (
    TRACK_ARTIST_MAX_LENGTH,
    TRACK_TITLE_MAX_LENGTH,
    TrackStatus,
)

Title = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=TRACK_TITLE_MAX_LENGTH,
    ),
]
Artist = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=TRACK_ARTIST_MAX_LENGTH,
    ),
]


class TrackResponse(BaseModel):
    id: UUID
    """ID of the track."""
    title: str
    """Title of the track."""
    artist: str | None
    """Artist of the track."""
    duration_ms: int
    """Duration of the track in milliseconds."""
    status: TrackStatus
    """Status of the track."""
    created_at: datetime
    """Creation time of the track."""
    updated_at: datetime
    """Last update time of the track."""
    removed_at: datetime | None
    """Removal time of the track, if applicable."""


class TrackListResponse(BaseModel):
    items: list[TrackResponse]
    """List of tracks in the current page."""
    next_cursor: str | None
    """Cursor for the next page of tracks, if applicable."""


class UpdateTrackRequest(BaseModel):
    title: Title | None = None
    """New title of the track, if updating."""
    artist: Artist | None = None
    """New artist of the track, if updating."""

    @field_validator("title")
    @classmethod
    def title_not_null(cls, title: str | None) -> str | None:
        # Omitting title is fine; an explicit null would break the NOT NULL column.
        if title is None:
            raise ValueError("title cannot be null")
        return title


class PlaybackUrlResponse(BaseModel):
    url: str
    """Presigned URL to stream the audio."""
    expires_at: datetime
    """Expiration time of the presigned URL."""
