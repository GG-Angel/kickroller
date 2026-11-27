from unittest.mock import AsyncMock

import pytest
from faststream.rabbit import RabbitBroker, TestRabbitBroker

from kickroller.broker import start_broker

pytestmark = pytest.mark.anyio


async def test_start_broker_starts_and_logs_the_host_without_credentials(
    logs: list[str],
) -> None:
    broker = RabbitBroker("amqp://user:secret@rabbitmq:5672/")

    async with TestRabbitBroker(broker, connect_only=True):
        await start_broker(broker)
        start = broker.start
        assert isinstance(start, AsyncMock)
        start.assert_awaited_once()

    assert logs == [
        "start_broker INFO Connected to RabbitMQ at amqp://rabbitmq:5672/\n"
    ]
