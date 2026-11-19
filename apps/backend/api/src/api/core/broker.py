from fastapi import Depends
from faststream.rabbit import RabbitBroker
from loguru import logger

from api.consumers import storage
from api.core.config import BrokerConfig
from api.core.state import AppState, get_app_state


def build_broker(config: BrokerConfig) -> RabbitBroker:
    """Build the RabbitMQ broker. Connects when the app starts."""
    broker = RabbitBroker(str(config.url))
    broker.include_router(storage.router)
    return broker


async def start_broker(broker: RabbitBroker) -> None:
    """Connect the broker and start its subscribers."""
    await broker.start()
    logger.info(
        f"Connected to RabbitMQ at {', '.join(broker.specification.url)}"
    )


def get_broker(state: AppState = Depends(get_app_state)) -> RabbitBroker:
    """FastAPI dependency that returns the shared broker for publishing."""
    return state.broker
