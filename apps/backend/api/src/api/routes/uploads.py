from fastapi import APIRouter, Depends, status

from api.routes.tracks import not_implemented
from api.schemas import (
    CreateUploadRequest,
    CreateUploadResponse,
)
from api.services.storage import ObjectStorage, get_object_storage

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_upload(
    body: CreateUploadRequest,
    storage: ObjectStorage = Depends(get_object_storage),
) -> CreateUploadResponse:
    """Make a pending upload and sign a PUT URL for it."""
    raise not_implemented()
