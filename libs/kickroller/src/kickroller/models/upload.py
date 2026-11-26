from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Enum, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from kickroller.models.base import Base

UPLOAD_FILENAME_MAX_LENGTH = 500


class UploadStatus(StrEnum):
    """Lifecycle state of an upload attempt."""

    PENDING = "pending"
    """URL issued, upload not confirmed."""
    VALIDATING = "validating"
    """Worker is examining the object."""
    REJECTED = "rejected"
    """Failed validation."""
    EXPIRED = "expired"
    """Upload never arrived."""


class Upload(Base):
    """One upload attempt. Short-lived: a cleanup job deletes old rows."""

    __tablename__ = "upload"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        server_default=func.uuidv7(),
    )
    """The upload's unique identifier in the DB."""
    user_id: Mapped[UUID] = mapped_column()
    """The ID of the uploader. Comes from the JWT subject claim."""

    object_key: Mapped[str] = mapped_column(Text, unique=True)
    """Where the audio goes in the object store."""
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    """Declared size of the audio in bytes."""
    original_filename: Mapped[str] = mapped_column(Text)
    """The original filename of the uploaded audio."""
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    """When the signed upload URL stops working."""

    status: Mapped[UploadStatus] = mapped_column(
        Enum(
            UploadStatus,
            native_enum=False,
            create_constraint=True,
            name="upload_status_check",
            length=None,
            values_callable=lambda e: [m.value for m in e],
        ),
        server_default=UploadStatus.PENDING,
    )
    """Lifecycle state of the upload."""
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    """Why validation rejected the upload."""
    validation_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    """When a worker last claimed the upload. Stale values mean a crash."""
    validation_attempts: Mapped[int] = mapped_column(server_default="0")
    """How many times a worker has claimed the upload."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    """The timestamp of when the upload was created."""
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    """The timestamp of the last update."""

    __table_args__ = (
        CheckConstraint(size_bytes > 0, name="upload_size_bytes_check"),
        CheckConstraint(
            (status != UploadStatus.VALIDATING)
            | validation_started_at.is_not(None),
            name="upload_validation_started_at_check",
        ),
        CheckConstraint(
            validation_attempts >= 0, name="upload_validation_attempts_check"
        ),
        CheckConstraint(
            func.length(original_filename).between(
                1, UPLOAD_FILENAME_MAX_LENGTH
            ),
            name="upload_original_filename_check",
        ),
    )
