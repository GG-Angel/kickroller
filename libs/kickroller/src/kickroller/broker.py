from faststream.rabbit import RabbitBroker

from kickroller.config import BrokerConfig


def build_broker(config: BrokerConfig) -> RabbitBroker:
    """Build the RabbitMQ broker. Connects when the app starts."""
    return RabbitBroker(str(config.url))
