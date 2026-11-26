from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from faststream import FastStream

from ingester.core.broker import build_broker
from ingester.core.config import Config
from ingester.core.state import AppState
from ingester.core.storage import create_store


@asynccontextmanager
async def open_state(config: Config) -> AsyncGenerator[AppState]:
    """Open the app's resources and release them on exit."""
    store = await create_store(config.storage)
    yield AppState(store=store, broker=build_broker(config.broker))


def create_app(state: AppState) -> FastStream:
    app = FastStream(state.broker)
    app.context.set_global("state", state)
    return app
