"""The storage event consumer, run through the app's real broker wiring."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from faststream.rabbit import TestRabbitBroker

from api.consumers.storage import UPLOADS_CREATED
from api.core.broker import build_broker
from api.core.config import BrokerConfig
from api.core.state import AppState
from api.server import create_asgi_app
from tests.fakes import load_settings

pytestmark = pytest.mark.anyio


def storage_event(name: str) -> dict[str, object]:
    record = {
        "eventName": name,
        "s3": {
            "bucket": {"name": "kickroller"},
            "object": {"key": "uploads/abc"},
        },
    }
    return {"EventName": name, "Records": [record]}


async def test_object_puts_reach_the_handler(
    state: AppState, logs: list[str]
) -> None:
    config = load_settings(BrokerConfig, url="amqp://rabbitmq:5672/")
    broker = build_broker(config)
    app = create_asgi_app(replace(state, broker=broker))

    async with TestRabbitBroker(broker, connect_only=True) as test_broker:
        with TestClient(app):
            await test_broker.publish(
                json.dumps(storage_event("s3:ObjectCreated:Put")),
                UPLOADS_CREATED,
                content_type="application/json",
            )

    seen = [line for line in logs if "handle_upload_created" in line]
    assert len(seen) == 1
    assert seen[0].startswith("handle_upload_created INFO Saw storage event")
