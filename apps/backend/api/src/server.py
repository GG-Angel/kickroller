from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import Config
from src.core.db import close_pool, create_pool
from src.core.logging import configure_logging
from src.core.state import AppState


def create_app(config: Config | None = None) -> FastAPI:
    """Build the API app. Reads settings from the environment if none are given."""
    config = config or Config()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        state = AppState(db_pool=await create_pool(config.database))
        app.state.app_state = state

        try:
            yield
        finally:
            await close_pool(state.db_pool)

    app = FastAPI(lifespan=lifespan)
    return app
