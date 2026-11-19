from fastapi import Depends
from faststream.rabbit import RabbitQueue, RabbitRouter
from loguru import logger

from api.schemas import StorageEvent
from api.services.upload import UploadService, get_upload_service

router = RabbitRouter()

# Owned by infra, so only consume, never declare.
UPLOADS_CREATED = RabbitQueue("uploads.created", declare=False)


@router.subscriber(UPLOADS_CREATED)
async def handle_upload_created(
    event: StorageEvent,
    uploads: UploadService = Depends(get_upload_service),
) -> None:
    # Implement the handling logic here
    logger.info(f"Saw storage event: {event}")
