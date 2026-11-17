from fastapi import APIRouter

from api.routes import health, tracks, uploads

router = APIRouter()
router.include_router(health.router)
router.include_router(tracks.router)
router.include_router(uploads.router)
