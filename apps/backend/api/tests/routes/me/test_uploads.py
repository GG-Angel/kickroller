"""HTTP contract of the upload routes. The upload service is mocked."""

from typing import Any
from unittest.mock import Mock
from uuid import uuid7

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.schemas import CreateUploadResponse
from api.services.upload import UploadService, get_upload_service
from tests.fakes import NOW, USER_ID

UPLOADS_URL = "/me/uploads"


@pytest.fixture
def uploads(app: FastAPI) -> Mock:
    uploads = Mock(spec=UploadService)
    app.dependency_overrides[get_upload_service] = lambda: uploads
    return uploads


def test_create_upload_returns_the_signed_upload(
    client: TestClient, uploads: Mock
) -> None:
    created = CreateUploadResponse(
        upload_id=uuid7(), upload_url="http://rustfs/put", expires_at=NOW
    )
    uploads.create.return_value = created

    response = client.post(
        UPLOADS_URL, json={"filename": "kick.wav", "size_bytes": 1}
    )

    assert response.status_code == 201
    assert CreateUploadResponse.model_validate(response.json()) == created
    uploads.create.assert_awaited_once_with(USER_ID, "kick.wav")


@pytest.mark.parametrize("user_id", [None, "", "not-a-uuid"])
def test_create_upload_requires_a_user_id(
    client: TestClient, uploads: Mock, user_id: str | None
) -> None:
    del client.headers["X-User-Id"]
    headers = {} if user_id is None else {"X-User-Id": user_id}

    response = client.post(
        UPLOADS_URL,
        json={"filename": "kick.wav", "size_bytes": 1},
        headers=headers,
    )

    assert response.status_code == 422
    uploads.create.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"filename": "", "size_bytes": 1},
        {"filename": "x" * 501, "size_bytes": 1},
        {"filename": "kick.wav", "size_bytes": 0},
        {"filename": "kick.wav"},
    ],
    ids=["empty-name", "long-name", "zero-size", "no-size"],
)
def test_create_upload_rejects_invalid_bodies(
    client: TestClient, uploads: Mock, body: dict[str, Any]
) -> None:
    assert client.post(UPLOADS_URL, json=body).status_code == 422
    uploads.create.assert_not_called()
