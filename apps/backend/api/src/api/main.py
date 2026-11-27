import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from faststream_fastapi import AsyncAPIConfig, FastStreamAPI
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.config import Config
from api.core.state import AppState
from api.routes import health
from api.routes.v1.router import user_router as v1_router
from kickroller.broker import build_broker, start_broker
from kickroller.db import open_db
from kickroller.logging import configure_logging
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
    app.include_router(health.router)  # Unversioned, for the healthcheck.
    app.include_router(v1_router, prefix="/v1")
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


async def main() -> None:
    config = Config()
    configure_logging()
    async with open_state(config) as state:
        await serve(config, state)  # share db pool, storage, and broker


async def serve(config: Config, state: AppState) -> None:
    """Run the API and metrics servers until shutdown."""
    api_server = uvicorn.Server(
        uvicorn.Config(
            app=create_app_with_broker(
                state, serve_docs=config.server.serve_docs
            ),
            host=str(config.server.host),
            port=config.server.port,
            log_config=None,  # Handled by loguru
        )
    )
    metrics_server = uvicorn.Server(
        uvicorn.Config(
            app=make_asgi_app(),
            host=str(config.server.host),
            port=config.server.metrics_port,
            log_config=None,  # Handled by loguru
            lifespan="off",  # Has no lifespan; avoids uvicorn's "unsupported" log
            access_log=False,  # Scrapes would flood the logs
        )
    )
    await asyncio.gather(api_server.serve(), metrics_server.serve())


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
