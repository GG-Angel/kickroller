from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.config import Config
from api.core.db import open_db
from api.core.state import AppState
from api.core.storage import build_store, create_store
from api.routes import router
from api.services.storage import ObjectStorage


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    async with open_db(config.database) as db:
        yield AppState(
            db=db,
            storage=ObjectStorage(
                await create_store(config.storage),
                build_store(
                    config.storage,
                    config.storage.public_endpoint or config.storage.endpoint,
                ),
                config.storage.upload_url_ttl,
                config.storage.playback_url_ttl,
            ),
        )


def create_app(state: AppState) -> FastAPI:
    """Build the public API app."""
    app = FastAPI()
    app.state.app_state = state
    Instrumentator().instrument(app)
    app.include_router(router)
    return app
