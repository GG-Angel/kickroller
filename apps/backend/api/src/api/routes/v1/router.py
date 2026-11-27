from fastapi import APIRouter, Depends

from api.core.auth import get_user_id
from api.routes.me import tracks, uploads

# Everything under /me acts as the caller, so every route needs a user id.
router = APIRouter(prefix="/me", dependencies=[Depends(get_user_id)])
router.include_router(tracks.router)
router.include_router(uploads.router)
