from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from faststream_fastapi import AsyncAPIConfig, FastStreamAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.broker import build_broker, start_broker
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


def create_app(state: AppState, *, docs: bool = False) -> FastAPI:
    """Build the public API app."""
    # Without the schema URL, FastAPI also drops /docs and /redoc.
    app = FastAPI(openapi_url="/openapi.json" if docs else None)
    app.state.app_state = state
    Instrumentator().instrument(app)
    app.include_router(router)
    return app


def create_asgi_app(state: AppState, *, docs: bool = False) -> FastStreamAPI:
    """The API with the broker connected and closed in its lifespan."""
    return FastStreamAPI(
        state.broker,
        application=create_app(state, docs=docs),
        safe_connection=True,  # Fail startup if RabbitMQ is down.
        connection_hook=start_broker,
        # Serves /asyncapi, .json and .yaml. No "try it out": it would let any
        # caller publish to the broker.
        asyncapi_path=(
            AsyncAPIConfig("/asyncapi", try_it_out_path=None) if docs else None
        ),
    )
