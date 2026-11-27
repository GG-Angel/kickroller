from fastapi import APIRouter, Depends

from api.routes.dependencies import get_user_id
from api.routes.v1.endpoints import tracks, uploads

user_router = APIRouter(prefix="/me", dependencies=[Depends(get_user_id)])
user_router.include_router(tracks.router)
user_router.include_router(uploads.router)
