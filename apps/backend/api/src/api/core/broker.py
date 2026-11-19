from faststream.rabbit.fastapi import RabbitRouter

from api.core.config import BrokerConfig


def build_broker(config: BrokerConfig) -> RabbitRouter:
    """Build a RabbitRouter for the given broker configuration."""
    return RabbitRouter(str(config.url))
