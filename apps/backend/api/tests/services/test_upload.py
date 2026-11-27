"""Unit tests for UploadService. The repository and storage are mocked."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from api.db.upload_repository import UploadDbRepository
from api.services.upload_service import UploadService
from kickroller.models import Upload
from tests.fakes import USER_ID

pytestmark = pytest.mark.anyio


async def test_create_adds_pending_upload_and_signs_post_form(
    storage: Mock,
) -> None:
    repo = Mock(spec=UploadDbRepository)
    before = datetime.now(UTC)

    response = await UploadService(repo, storage).create(USER_ID, "kick.wav")

    upload: Upload = repo.add.call_args.args[0]
    assert upload.id == response.upload_id
    assert upload.user_id == USER_ID
    assert upload.object_key == f"uploads/{upload.id}"
    assert upload.filename == "kick.wav"
    assert upload.expires_at == response.expires_at
    assert before + storage.upload_url_ttl <= upload.expires_at
    assert response.upload_url == "http://rustfs/bucket"
    assert response.upload_fields == {"key": "uploads/a"}
    storage.create_post_form.assert_awaited_once_with(
        key=upload.object_key, expires_in=storage.upload_url_ttl
    )
