from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Index, func
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class Track(Base):
    """A track a user uploaded. The audio itself lives in the object store."""

    __tablename__ = "track"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=func.uuidv7(),
    )
    """The track's unique identifier in the DB."""
    user_id: Mapped[str] = mapped_column()
    """The ID of the uploader. Comes from the JWT subject claim."""

    name: Mapped[str] = mapped_column()
    """The name of the track."""
    audio_id: Mapped[str] = mapped_column(unique=True)
    """Object store key of the audio file."""

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

    __table_args__ = (
        Index("track_user_id_created_at_idx", user_id, created_at.desc()),
    )
