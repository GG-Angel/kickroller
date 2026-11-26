from dataclasses import dataclass

from faststream import Context
from faststream.rabbit import RabbitBroker
from obstore.store import S3Store
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created at startup."""

    db_engine: AsyncEngine
    store: S3Store
    broker: RabbitBroker


def get_app_state(state: AppState = Context()) -> AppState:
    """FastStream dependency that returns the typed app state."""
    return state
