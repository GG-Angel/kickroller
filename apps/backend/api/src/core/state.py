from dataclasses import dataclass

import asyncpg
from fastapi import Request
from obstore.store import S3Store


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created in the lifespan."""

    db_pool: asyncpg.Pool
    store: S3Store


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency that returns the typed app state."""
    return request.app.state.app_state
