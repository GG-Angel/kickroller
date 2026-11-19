"""Unit tests for building the app. Nothing here needs Postgres or RustFS."""

from dataclasses import replace
from typing import Annotated
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from faststream.rabbit import RabbitBroker, TestRabbitBroker

from api.core.state import AppState
from api.server import create_app, create_asgi_app


def test_app_holds_the_given_state(state: AppState) -> None:
    assert create_app(state).state.app_state is state


def test_metrics_are_not_served_on_the_api(state: AppState) -> None:
    with TestClient(create_app(state)) as client:
        assert client.get("/metrics").status_code == 404


@pytest.mark.anyio
async def test_lifespan_starts_broker_and_runs_handlers_with_fastapi_di(
    state: AppState,
) -> None:
    broker = RabbitBroker()
    handled = Mock()

    @broker.subscriber("test")
    async def handle(
        body: str, dep: Annotated[int, Depends(lambda: 7)]
    ) -> None:
        handled(body, dep)

    app = create_asgi_app(replace(state, broker=broker))
    # connect_only: the app's lifespan, not the test broker, starts it.
    async with TestRabbitBroker(broker, connect_only=True) as test_broker:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            await test_broker.publish("hi", "test")
            handled.assert_called_once_with("hi", 7)

        stop = broker.stop
        assert isinstance(stop, AsyncMock)
        stop.assert_awaited_once()


def test_startup_fails_when_rabbitmq_is_unreachable(
    state: AppState,
) -> None:
    broker = RabbitBroker("amqp://guest:guest@127.0.0.1:1/")  # Nothing listens.
    app = create_asgi_app(replace(state, broker=broker))

    with pytest.raises(ConnectionError), TestClient(app):
        pass
