from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.core.auth import get_user_id
from api.schemas import (
    CreateUploadRequest,
    CreateUploadResponse,
)
from api.services.upload import UploadService, get_upload_service

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_upload(
    body: CreateUploadRequest,
    user_id: UUID = Depends(get_user_id),
    uploads: UploadService = Depends(get_upload_service),
) -> CreateUploadResponse:
    """Make a pending upload and sign a PUT URL for it."""
    return await uploads.create(user_id, body.filename)
