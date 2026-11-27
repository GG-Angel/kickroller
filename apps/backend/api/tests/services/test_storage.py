"""Unit tests for StorageService. Signing is local, so nothing connects."""

import base64
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from types_aiobotocore_s3 import S3Client

from api.core.config import StorageConfig
from api.services.storage import StorageService
from kickroller.models import UPLOAD_MAX_SIZE_BYTES
from kickroller.storage import open_store
from tests.fakes import load_settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def store() -> Mock:
    return Mock(spec=S3Client)


@pytest.fixture
async def storage(store: Mock) -> AsyncIterator[StorageService]:
    config = load_settings(
        StorageConfig,
        endpoint="http://rustfs:9000",
        bucket="bucket",
        access_key_id="key",
        secret_access_key="secret",
    )
    async with open_store(config, check_bucket_access=False) as signer:
        yield StorageService(
            store, signer, "bucket", timedelta(minutes=15), timedelta(hours=1)
        )


async def test_get_url_is_signed_for_the_key_and_expiry(
    storage: StorageService,
) -> None:
    url = await storage.create_get_url("tracks/a", timedelta(minutes=5))

    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
        "http://rustfs:9000/bucket/tracks/a"
    )
    assert parse_qs(parts.query)["X-Amz-Expires"] == ["300"]


async def test_post_form_policy_caps_size_and_requires_audio(
    storage: StorageService,
) -> None:
    before = datetime.now(UTC).replace(microsecond=0)

    form = await storage.create_post_form("uploads/a", timedelta(minutes=5))

    assert form.url == "http://rustfs:9000/bucket"
    assert form.fields["key"] == "uploads/a"
    policy = json.loads(base64.b64decode(form.fields["policy"]))
    assert {"key": "uploads/a"} in policy["conditions"]
    assert ["content-length-range", 1, UPLOAD_MAX_SIZE_BYTES] in (
        policy["conditions"]
    )
    assert ["starts-with", "$Content-Type", "audio/"] in policy["conditions"]
    expiration = datetime.fromisoformat(policy["expiration"])
    assert before + timedelta(minutes=5) <= expiration


async def test_delete_removes_the_object_from_the_bucket(
    storage: StorageService, store: Mock
) -> None:
    await storage.delete("uploads/a")

    store.delete_object.assert_awaited_once_with(
        Bucket="bucket", Key="uploads/a"
    )
