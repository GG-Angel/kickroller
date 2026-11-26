from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from faststream import FastStream

from ingester.consumers import uploads
from ingester.core.config import Config
from ingester.core.state import AppState
from kickroller.broker import build_broker
from kickroller.db import open_db
from kickroller.storage import open_store


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    async with (
        open_db(config.database) as db_engine,
        open_store(config.storage) as store,
    ):
        yield AppState(
            db_engine=db_engine,
            store=store,
            broker=build_broker(config.broker),
        )


def create_app(state: AppState) -> FastStream:
    state.broker.include_router(uploads.router)
    app = FastStream(state.broker)
    app.context.set_global("state", state)
    return app
