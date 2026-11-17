from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from api.models import UPLOAD_FILENAME_MAX_LENGTH


class CreateUploadRequest(BaseModel):
    filename: Annotated[
        str, Field(min_length=1, max_length=UPLOAD_FILENAME_MAX_LENGTH)
    ]
    size: Annotated[int, Field(gt=0)]
    """Size of the audio in bytes."""


class CreateUploadResponse(BaseModel):
    upload_id: UUID
    upload_url: str
    expires_at: datetime
