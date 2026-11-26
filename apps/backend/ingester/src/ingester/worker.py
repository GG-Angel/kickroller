from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from faststream import FastStream

from ingester.core.config import Config
from ingester.core.state import AppState
from kickroller.broker import build_broker
from kickroller.db import open_db
from kickroller.storage import create_store


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    async with open_db(config.database) as db_engine:
        store = await create_store(config.storage)
        yield AppState(
            db_engine=db_engine,
            store=store,
            broker=build_broker(config.broker),
        )


def create_app(state: AppState) -> FastStream:
    app = FastStream(state.broker)
    app.context.set_global("state", state)
    return app
