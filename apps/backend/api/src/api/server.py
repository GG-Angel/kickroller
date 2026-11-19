from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from faststream_fastapi import FastStreamAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.broker import build_broker
from api.core.config import Config
from api.core.db import open_db
from api.core.state import AppState
from api.core.storage import build_object_storage, create_store
from api.routes import router


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    async with open_db(config.database) as db:
        store = await create_store(config.storage)
        yield AppState(
            db=db,
            storage=build_object_storage(config.storage, store),
            broker=build_broker(config.broker),
        )


def create_app(state: AppState) -> FastAPI:
    """Build the public API app."""
    app = FastAPI()
    app.state.app_state = state
    Instrumentator().instrument(app)
    app.include_router(router)
    return app


def create_asgi_app(state: AppState) -> FastStreamAPI:
    """The API with the broker connected and closed in its lifespan."""
    return FastStreamAPI(
        state.broker,
        application=create_app(state),
        safe_connection=True,  # Fail startup if RabbitMQ is down.
    )
