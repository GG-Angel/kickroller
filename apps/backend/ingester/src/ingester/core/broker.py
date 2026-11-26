from faststream.rabbit import RabbitBroker

from ingester.core.config import BrokerConfig


def build_broker(config: BrokerConfig) -> RabbitBroker:
    """Build the RabbitMQ broker. Connects when the app starts."""
    return RabbitBroker(str(config.url))
