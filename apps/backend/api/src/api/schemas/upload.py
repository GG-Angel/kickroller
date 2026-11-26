from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from kickroller.models import UPLOAD_FILENAME_MAX_LENGTH


class CreateUploadRequest(BaseModel):
    filename: Annotated[
        str, Field(min_length=1, max_length=UPLOAD_FILENAME_MAX_LENGTH)
    ]
    """Original filename of the audio."""


class CreateUploadResponse(BaseModel):
    upload_id: UUID
    """ID of the created upload."""
    upload_url: str
    """Presigned URL to upload the audio."""
    expires_at: datetime
    """Expiration time of the presigned URL."""
