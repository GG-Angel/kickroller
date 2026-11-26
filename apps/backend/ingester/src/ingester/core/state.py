from dataclasses import dataclass

from faststream import Context
from faststream.rabbit import RabbitBroker
from obstore.store import S3Store


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created at startup."""

    store: S3Store
    broker: RabbitBroker


def get_app_state(state: AppState = Context()) -> AppState:
    """FastStream dependency that returns the typed app state."""
    return state
