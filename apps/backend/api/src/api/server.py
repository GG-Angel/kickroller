from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from faststream_fastapi import AsyncAPIConfig, FastStreamAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.broker import start_broker
from api.core.config import Config
from api.core.state import AppState
from api.routes import router
from kickroller.broker import build_broker
from kickroller.db import open_db
from kickroller.storage import open_store


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    async with (
        open_db(config.database) as db_engine,
        open_store(config.storage) as store,
        open_store(
            config.storage.with_public_endpoint(),
            check_bucket_access=False,
        ) as url_signing_store,
    ):
        yield AppState(
            config=config,
            db_engine=db_engine,
            store=store,
            url_signing_store=url_signing_store,
            broker=build_broker(config.broker),
        )


def create_http_app(state: AppState, *, serve_docs: bool = False) -> FastAPI:
    """Build the public API app."""
    app = FastAPI(openapi_url="/openapi.json" if serve_docs else None)
    app.state.app_state = state
    Instrumentator().instrument(app)
    app.include_router(router)
    return app


def create_app_with_broker(
    state: AppState, *, serve_docs: bool = False
) -> FastStreamAPI:
    """The API with the broker connected and closed in its lifespan."""
    return FastStreamAPI(
        state.broker,
        application=create_http_app(state, serve_docs=serve_docs),
        safe_connection=True,  # Fail startup if RabbitMQ is down.
        connection_hook=start_broker,
        asyncapi_path=(
            AsyncAPIConfig("/asyncapi", try_it_out_path=None)
            if serve_docs
            else None
        ),
    )
