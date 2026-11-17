"""Unit tests for API startup and shutdown in create_app.

The state factory is mocked, so these run without Postgres or RustFS.
"""

from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from api.core.config import Config
from api.core.state import AppState
from api.server import create_app
from api.services.storage import ObjectStorage


@pytest.fixture
def config() -> Mock:
    return Mock(spec=Config)


@pytest.fixture
def state() -> AppState:
    return AppState(db=Mock(spec=AsyncEngine), storage=Mock(spec=ObjectStorage))


@pytest.fixture
def state_factory(state: AppState) -> MagicMock:
    """Stands in for open_state: an async context manager yielding the state."""
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = state
    return factory


def test_startup_opens_state_and_shutdown_releases_it(
    config: Mock, state: AppState, state_factory: MagicMock
) -> None:
    app = create_app(config, state_factory)
    resources = state_factory.return_value

    with TestClient(app):
        state_factory.assert_called_once_with(config)
        assert app.state.app_state is state
        resources.__aexit__.assert_not_awaited()

    resources.__aexit__.assert_awaited_once()


def test_startup_fails_when_resources_fail_to_open(
    config: Mock, state_factory: MagicMock
) -> None:
    state_factory.return_value.__aenter__.side_effect = OSError("rustfs down")
    app = create_app(config, state_factory)

    with pytest.raises(OSError, match="rustfs down"), TestClient(app):
        pass


def test_health_returns_ok(config: Mock, state_factory: MagicMock) -> None:
    with TestClient(create_app(config, state_factory)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_are_not_served_on_the_api(
    config: Mock, state_factory: MagicMock
) -> None:
    with TestClient(create_app(config, state_factory)) as client:
        assert client.get("/metrics").status_code == 404
