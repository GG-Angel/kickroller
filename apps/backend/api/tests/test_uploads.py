"""Unit tests for the upload routes. Session and storage are mocked."""

from collections.abc import Iterator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from api.core.config import Config
from api.core.db import get_session
from api.models import Upload, UploadStatus
from api.server import create_app
from api.services.storage import ObjectStorage, get_object_storage

UPLOAD_ID = UUID("01900000-0000-7000-8000-000000000001")
USER_ID = UUID("01900000-0000-7000-8000-0000000000aa")


@pytest.fixture
def session() -> AsyncMock:
    session = AsyncMock()
    session.add = Mock()
    session.execute.return_value.scalar_one = Mock(return_value=UPLOAD_ID)
    return session


@pytest.fixture
def storage() -> Mock:
    storage = Mock(spec=ObjectStorage)
    storage.sign_put = AsyncMock(return_value="http://rustfs/signed")
    return storage


@pytest.fixture
def client(session: AsyncMock, storage: Mock) -> Iterator[TestClient]:
    factory = MagicMock()
    app = create_app(Mock(spec=Config), factory)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_object_storage] = lambda: storage
    with TestClient(app) as client:
        yield client


def test_create_upload_stores_pending_upload_and_signs_put(
    client: TestClient, session: AsyncMock, storage: Mock
) -> None:
    response = client.post(
        "/uploads",
        json={"filename": "kick.wav", "size": 1234},
        headers={"X-User-Id": str(USER_ID)},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["upload_id"] == str(UPLOAD_ID)
    assert body["upload_url"] == "http://rustfs/signed"

    upload: Upload = session.add.call_args.args[0]
    assert upload.id == UPLOAD_ID
    assert upload.user_id == USER_ID
    assert upload.object_key == f"uploads/{UPLOAD_ID}"
    assert upload.size == 1234
    assert upload.original_filename == "kick.wav"
    assert upload.expires_at == datetime.fromisoformat(body["expires_at"])
    assert upload.status in (None, UploadStatus.PENDING)  # server default
    session.commit.assert_awaited_once()
    storage.sign_put.assert_awaited_once()
    assert storage.sign_put.call_args.args[0] == upload.object_key


def test_create_upload_requires_user(
    client: TestClient, session: AsyncMock
) -> None:
    response = client.post("/uploads", json={"filename": "kick.wav", "size": 1})

    assert response.status_code == 422
    session.add.assert_not_called()
