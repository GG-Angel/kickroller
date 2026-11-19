"""Unit tests for create_app. The state is mocked, so no Postgres or RustFS."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from api.core.state import AppState
from api.server import create_app
from api.services.storage import ObjectStorage


@pytest.fixture
def state() -> AppState:
    return AppState(db=Mock(spec=AsyncEngine), storage=Mock(spec=ObjectStorage))


def test_app_holds_the_given_state(state: AppState) -> None:
    assert create_app(state).state.app_state is state


def test_health_returns_ok(state: AppState) -> None:
    with TestClient(create_app(state)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_are_not_served_on_the_api(state: AppState) -> None:
    with TestClient(create_app(state)) as client:
        assert client.get("/metrics").status_code == 404
