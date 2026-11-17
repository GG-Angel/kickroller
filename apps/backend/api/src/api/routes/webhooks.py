from fastapi import APIRouter, status

from api.routes.tracks import not_implemented
from api.schemas import StorageEvent

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/storage", status_code=status.HTTP_204_NO_CONTENT)
async def receive_storage_event(body: StorageEvent) -> None:
    """Mark uploads as uploaded when RustFS reports their objects were put.

    Must check the webhook token, URL-decode each key, and be idempotent,
    since RustFS can deliver the same event more than once.
    """
    raise not_implemented()
