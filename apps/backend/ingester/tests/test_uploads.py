"""The uploads.created consumer, run through the app's broker wiring."""

from typing import Any
from unittest.mock import Mock

import pytest
from faststream.rabbit import RabbitBroker, TestRabbitBroker
from sqlalchemy.ext.asyncio import AsyncEngine
from types_aiobotocore_s3 import S3Client

from ingester.consumers.uploads import (
    UPLOADS_CREATED,
    handle_upload_created,
)
from ingester.core.state import AppState
from ingester.worker import create_app

pytestmark = pytest.mark.anyio


def storage_event(*keys: str) -> dict[str, Any]:
    records = [
        {"eventName": "s3:ObjectCreated:Put", "s3": {"object": {"key": key}}}
        for key in keys
    ]
    return {"EventName": "s3:ObjectCreated:Put", "Records": records}


async def test_upload_events_reach_the_handler() -> None:
    broker = RabbitBroker()
    create_app(
        AppState(
            db_engine=Mock(spec=AsyncEngine),
            store=Mock(spec=S3Client),
            broker=broker,
        )
    )
    event = storage_event("uploads/abc")

    async with TestRabbitBroker(broker) as test_broker:
        await test_broker.publish(event, UPLOADS_CREATED)
        handle_upload_created.mock.assert_called_once_with(event)
