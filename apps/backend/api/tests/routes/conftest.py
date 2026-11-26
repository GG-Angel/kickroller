"""Route fixtures. Tests swap services in through dependency_overrides."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.core.state import AppState
from api.server import create_http_app
from tests.fakes import USER_ID


@pytest.fixture
def app(state: AppState) -> FastAPI:
    return create_http_app(state)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers={"X-User-Id": str(USER_ID)}) as client:
        yield client
