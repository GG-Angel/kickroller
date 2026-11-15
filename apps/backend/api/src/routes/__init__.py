from fastapi import APIRouter

from src.routes import health

router = APIRouter()
router.include_router(health.router)
