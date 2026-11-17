from fastapi import APIRouter

from api.routes import health, tracks

router = APIRouter()
router.include_router(health.router)
router.include_router(tracks.router)
