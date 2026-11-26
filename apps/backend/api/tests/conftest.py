"""Fixtures shared across the API tests."""

from collections.abc import Iterator
from datetime import timedelta
from unittest.mock import Mock

import pytest
from faststream.rabbit import RabbitBroker
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine
from types_aiobotocore_s3 import S3Client

from api.core.config import Config, StorageConfig
from api.core.state import AppState
from api.services.storage import StorageService
from tests.fakes import load_settings


@pytest.fixture
def storage() -> Mock:
    """Object storage that signs fake URLs. Its async methods are AsyncMocks."""
    storage = Mock(spec=StorageService)
    storage.upload_url_ttl = timedelta(minutes=15)
    storage.playback_url_ttl = timedelta(hours=1)
    storage.create_put_url.return_value = "http://rustfs/put"
    storage.create_get_url.return_value = "http://rustfs/get"
    return storage


@pytest.fixture
def state() -> AppState:
    """App state with mocked resources. The broker never connects."""
    config = Mock(spec=Config)
    config.storage = load_settings(
        StorageConfig, access_key_id="key", secret_access_key="secret"
    )
    return AppState(
        config=config,
        db_engine=Mock(spec=AsyncEngine),
        store=Mock(spec=S3Client),
        url_signing_store=Mock(spec=S3Client),
        broker=Mock(spec=RabbitBroker),
    )


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Loguru output, as '<function> <level> <message>' lines."""
    logs: list[str] = []
    sink = logger.add(logs.append, format="{function} {level} {message}")
    yield logs
    logger.remove(sink)
