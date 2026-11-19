from fastapi import Depends
from faststream.rabbit import RabbitBroker

from api.core.config import BrokerConfig
from api.core.state import AppState, get_app_state


def build_broker(config: BrokerConfig) -> RabbitBroker:
    """Build the RabbitMQ broker. Connects when the app starts."""
    return RabbitBroker(str(config.url))


def get_broker(state: AppState = Depends(get_app_state)) -> RabbitBroker:
    """FastAPI dependency that returns the shared broker for publishing."""
    return state.broker
