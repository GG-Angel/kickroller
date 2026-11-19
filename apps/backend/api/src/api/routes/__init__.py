from fastapi import APIRouter

from api.routes import health, me

router = APIRouter()
router.include_router(health.router)
router.include_router(me.router)
