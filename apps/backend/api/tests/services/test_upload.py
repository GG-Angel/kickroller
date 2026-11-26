"""Unit tests for UploadService. The repository and storage are mocked."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from api.repositories.upload import UploadRepository
from api.services.upload import UploadService
from kickroller.models import Upload
from tests.fakes import USER_ID

pytestmark = pytest.mark.anyio


async def test_create_adds_pending_upload_and_signs_put(storage: Mock) -> None:
    repo = Mock(spec=UploadRepository)
    before = datetime.now(UTC)

    response = await UploadService(repo, storage).create(
        USER_ID, "kick.wav", 1234
    )

    upload: Upload = repo.add.call_args.args[0]
    assert upload.id == response.upload_id
    assert upload.user_id == USER_ID
    assert upload.object_key == f"uploads/{upload.id}"
    assert upload.size_bytes == 1234
    assert upload.original_filename == "kick.wav"
    assert upload.expires_at == response.expires_at
    assert before + storage.upload_url_ttl <= upload.expires_at
    assert response.upload_url == "http://rustfs/put"
    storage.sign_put.assert_awaited_once_with(
        key=upload.object_key, expires_in=storage.upload_url_ttl
    )
