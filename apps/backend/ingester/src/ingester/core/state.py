from dataclasses import dataclass

from faststream import Context
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import AsyncEngine
from types_aiobotocore_s3 import S3Client


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created at startup."""

    db_engine: AsyncEngine
    store: S3Client
    broker: RabbitBroker


def get_app_state(state: AppState = Context()) -> AppState:
    """FastStream dependency that returns the typed app state."""
    return state
