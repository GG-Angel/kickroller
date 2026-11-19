"""Unit tests for create_app. The state is mocked, so no Postgres or RustFS."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from api.core.state import AppState
from api.server import create_app


def test_app_holds_the_given_state(storage: Mock) -> None:
    state = AppState(db=Mock(spec=AsyncEngine), storage=storage)

    assert create_app(state).state.app_state is state


def test_metrics_are_not_served_on_the_api(storage: Mock) -> None:
    app = create_app(AppState(db=Mock(spec=AsyncEngine), storage=storage))

    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404
