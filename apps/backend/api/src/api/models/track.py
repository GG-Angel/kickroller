from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Enum, Index, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base

TRACK_NAME_MAX_LENGTH = 300
TRACK_ARTIST_MAX_LENGTH = 300
TRACK_FILENAME_MAX_LENGTH = 500


class TrackStatus(StrEnum):
    """Lifecycle state of a track upload."""

    PENDING = "pending"
    """URL issued, upload not confirmed."""
    UPLOADED = "uploaded"
    """HEAD check passed, job queued."""
    VALIDATING = "validating"
    """Worker is examining the object."""
    READY = "ready"
    """Visible to users."""
    REJECTED = "rejected"
    """Failed validation."""
    EXPIRED = "expired"
    """Upload never arrived."""
    REMOVED = "removed"
    """Deleted by user or moderation."""


class Track(Base):
    """A track a user uploaded. The audio itself lives in the object store."""

    __tablename__ = "track"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=func.uuidv7(),
    )
    """The track's unique identifier in the DB."""
    user_id: Mapped[UUID] = mapped_column()
    """The ID of the uploader. Comes from the JWT subject claim."""

    name: Mapped[str] = mapped_column(Text)
    """The name of the track."""
    artist: Mapped[str | None] = mapped_column(Text)
    """The name of the artist."""
    original_filename: Mapped[str] = mapped_column(Text)
    """The original filename of the uploaded audio."""

    status: Mapped[TrackStatus] = mapped_column(
        Enum(
            TrackStatus,
            native_enum=False,
            create_constraint=True,
            name="track_status_check",
            length=None,
            values_callable=lambda e: [m.value for m in e],
        ),
        server_default=TrackStatus.PENDING,
    )
    """Lifecycle state of the upload."""
    rejection_reason: Mapped[str | None] = mapped_column()
    """Why validation rejected the track."""

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
    uploaded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    """When audio is confirmed to be in the object store."""
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    """When validation finished, whether the track was accepted or rejected."""
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    """When the track was removed."""

    __table_args__ = (
        Index("track_user_id_created_at_idx", user_id, created_at.desc()),
        CheckConstraint(
            func.length(name).between(1, TRACK_NAME_MAX_LENGTH),
            name="track_name_check",
        ),
        CheckConstraint(
            func.length(artist) <= TRACK_ARTIST_MAX_LENGTH,
            name="track_artist_check",
        ),
        CheckConstraint(
            func.length(original_filename) <= TRACK_FILENAME_MAX_LENGTH,
            name="track_original_filename_check",
        ),
    )
