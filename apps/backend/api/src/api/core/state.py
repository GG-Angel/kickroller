from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine

if TYPE_CHECKING:
    from api.services.storage import ObjectStorage


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created in the lifespan."""

    db: AsyncEngine
    storage: "ObjectStorage"


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency that returns the typed app state."""
    return request.app.state.app_state
