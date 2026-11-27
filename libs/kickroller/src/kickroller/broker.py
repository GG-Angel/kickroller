from faststream.rabbit import RabbitBroker
from loguru import logger

from kickroller.config import BrokerConfig


def build_broker(config: BrokerConfig) -> RabbitBroker:
    """Build the RabbitMQ broker. Connects when the app starts."""
    return RabbitBroker(str(config.url))


async def start_broker(broker: RabbitBroker) -> None:
    """Connect the broker and start its subscribers."""
    await broker.start()
    logger.info(
        f"Connected to RabbitMQ at {', '.join(broker.specification.url)}"
    )
