from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.config import Config
from api.core.db import open_db_pool
from api.core.logging import configure_logging
from api.core.state import AppState
from api.core.storage import create_store
from api.routes import router

type StateFactory = Callable[[Config], AbstractAsyncContextManager[AppState]]


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    async with open_db_pool(config.database) as db_pool:
        yield AppState(
            db_pool=db_pool,
            store=await create_store(config.storage),
        )


def create_app(
    config: Config,
    state_factory: StateFactory = open_state,
) -> FastAPI:
    """Build the API app."""
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with state_factory(config) as state:
            app.state.app_state = state
            yield

    app = FastAPI(lifespan=lifespan)
    Instrumentator().instrument(app)
    app.include_router(router)
    return app
