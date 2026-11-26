"""Building the app. Nothing here connects to RabbitMQ or RustFS."""

from unittest.mock import Mock

import pytest
from faststream import Depends
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from obstore.store import S3Store

from ingester.core.state import AppState, get_app_state
from ingester.worker import create_app

pytestmark = pytest.mark.anyio


async def test_handlers_get_the_app_state() -> None:
    broker = RabbitBroker()
    state = AppState(store=Mock(spec=S3Store), broker=broker)
    seen: list[AppState] = []

    @broker.subscriber("test")
    async def handle(
        body: str, state: AppState = Depends(get_app_state)
    ) -> None:
        seen.append(state)

    create_app(state)
    async with TestRabbitBroker(broker) as test_broker:
        await test_broker.publish("hi", "test")

    assert seen == [state]
