from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Enum, Index, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base

TRACK_TITLE_MAX_LENGTH = 300
TRACK_ARTIST_MAX_LENGTH = 300


class TrackStatus(StrEnum):
    """Lifecycle state of a validated track."""

    READY = "ready"
    """Visible to users."""
    REMOVED = "removed"
    """Deleted by user or moderation."""


class Track(Base):
    """A track that passed validation. The audio lives in the object store."""

    __tablename__ = "track"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=func.uuidv7(),
    )
    """The track's unique identifier in the DB."""
    user_id: Mapped[UUID] = mapped_column()
    """The ID of the uploader. Comes from the JWT subject claim."""

    title: Mapped[str] = mapped_column(Text)
    """The title of the track."""
    artist: Mapped[str | None] = mapped_column(Text)
    """The name of the artist."""

    object_key: Mapped[str] = mapped_column(Text, unique=True)
    """Where the audio lives in the object store."""
    duration_ms: Mapped[int] = mapped_column()
    """Length of the audio in milliseconds."""

    status: Mapped[TrackStatus] = mapped_column(
        Enum(
            TrackStatus,
            native_enum=False,
            create_constraint=True,
            name="track_status_check",
            length=None,
            values_callable=lambda e: [m.value for m in e],
        ),
        server_default=TrackStatus.READY,
    )
    """Whether the track is visible or removed."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    """The timestamp of when the track was created."""
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    """The timestamp of the last update."""
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    """When the track was removed."""

    __table_args__ = (
        Index("track_user_id_created_at_idx", user_id, created_at.desc()),
        CheckConstraint(
            func.length(title).between(1, TRACK_TITLE_MAX_LENGTH),
            name="track_title_check",
        ),
        CheckConstraint(
            func.length(artist) <= TRACK_ARTIST_MAX_LENGTH,
            name="track_artist_check",
        ),
        CheckConstraint(duration_ms > 0, name="track_duration_ms_check"),
    )
