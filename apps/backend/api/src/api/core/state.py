from dataclasses import dataclass

from fastapi import Request
from faststream.rabbit import RabbitBroker
from obstore.store import S3Store
from sqlalchemy.ext.asyncio import AsyncEngine

from api.core.config import Config


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created in the lifespan."""

    config: Config
    db: AsyncEngine
    store: S3Store
    signer: S3Store
    broker: RabbitBroker


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency that returns the typed app state."""
    return request.app.state.app_state
