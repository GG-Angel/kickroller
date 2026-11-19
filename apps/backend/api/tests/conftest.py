"""Fixtures shared across the API tests."""

from collections.abc import Iterator
from datetime import timedelta
from unittest.mock import Mock

import pytest
from faststream.rabbit import RabbitBroker
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine

from api.core.state import AppState
from api.services.storage import StorageService


@pytest.fixture
def storage() -> Mock:
    """Object storage that signs fake URLs. Its async methods are AsyncMocks."""
    storage = Mock(spec=StorageService)
    storage.upload_url_ttl = timedelta(minutes=15)
    storage.playback_url_ttl = timedelta(hours=1)
    storage.sign_put.return_value = "http://rustfs/put"
    storage.sign_get.return_value = "http://rustfs/get"
    return storage


@pytest.fixture
def state(storage: Mock) -> AppState:
    """App state with mocked resources. The broker never connects."""
    return AppState(
        db=Mock(spec=AsyncEngine),
        storage=storage,
        broker=Mock(spec=RabbitBroker),
    )


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Loguru output, as '<function> <level> <message>' lines."""
    logs: list[str] = []
    sink = logger.add(logs.append, format="{function} {level} {message}")
    yield logs
    logger.remove(sink)
