"""Unit tests for UploadRepository. The session is mocked, so no Postgres."""

from unittest.mock import Mock, call

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.repositories.upload import UploadRepository
from kickroller.models import Upload
from tests.fakes import NOW, USER_ID

pytestmark = pytest.mark.anyio


async def test_add_inserts_then_commits() -> None:
    session = Mock(spec=AsyncSession)
    upload = Upload(
        user_id=USER_ID,
        object_key="uploads/a",
        filename="kick.wav",
        expires_at=NOW,
    )

    await UploadRepository(session).add(upload)

    assert session.mock_calls == [call.add(upload), call.commit()]
