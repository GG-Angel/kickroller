from faststream.rabbit import RabbitQueue, RabbitRouter
from loguru import logger

from kickroller.schemas import StorageEvent

router = RabbitRouter()

UPLOADS_CREATED = RabbitQueue("uploads.created", declare=False)


@router.subscriber(UPLOADS_CREATED)
async def handle_upload_created(event: StorageEvent) -> None:
    for object_key in event.decode_object_keys():
        logger.info(f"Upload created: {object_key}")
