"""Unit tests for StorageService. Signing is local, so nothing connects."""

from collections.abc import AsyncIterator
from datetime import timedelta
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from types_aiobotocore_s3 import S3Client

from api.core.config import StorageConfig
from api.services.storage import StorageService
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


@pytest.mark.parametrize("method", ["create_put_url", "create_get_url"])
async def test_sign_returns_url_for_the_key_and_expiry(
    storage: StorageService, method: str
) -> None:
    url = await getattr(storage, method)("uploads/a", timedelta(minutes=5))

    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
        "http://rustfs:9000/bucket/uploads/a"
    )
    assert parse_qs(parts.query)["X-Amz-Expires"] == ["300"]


async def test_put_and_get_signatures_differ(storage: StorageService) -> None:
    expires_in = timedelta(minutes=5)

    put = await storage.create_put_url("uploads/a", expires_in)
    get = await storage.create_get_url("uploads/a", expires_in)

    assert put != get


async def test_delete_removes_the_object_from_the_bucket(
    storage: StorageService, store: Mock
) -> None:
    await storage.delete("uploads/a")

    store.delete_object.assert_awaited_once_with(
        Bucket="bucket", Key="uploads/a"
    )
