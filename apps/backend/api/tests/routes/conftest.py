"""Route fixtures. Tests swap services in through dependency_overrides."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from api.core.state import AppState
from api.server import create_app
from tests.fakes import USER_ID


@pytest.fixture
def app(storage: Mock) -> FastAPI:
    return create_app(AppState(db=Mock(spec=AsyncEngine), storage=storage))


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers={"X-User-Id": str(USER_ID)}) as client:
        yield client
