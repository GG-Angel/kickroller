"""Unit tests for ObjectStorage against in-memory and offline S3 stores."""

from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from obstore import list as list_objects
from obstore import put_async
from obstore.store import MemoryStore, S3Store

from api.services.storage import ObjectStorage

pytestmark = pytest.mark.anyio

# Signing is local, so this store never connects.
SIGNER = S3Store(
    "bucket",
    endpoint="http://rustfs:9000",
    access_key_id="key",
    secret_access_key="secret",
    region="us-east-1",
    client_options={"allow_http": True},
)


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def storage(store: MemoryStore) -> ObjectStorage:
    return ObjectStorage(
        store, SIGNER, timedelta(minutes=15), timedelta(hours=1)
    )


@pytest.mark.parametrize("method", ["sign_put", "sign_get"])
async def test_sign_returns_url_for_the_key_and_expiry(
    storage: ObjectStorage, method: str
) -> None:
    url = await getattr(storage, method)("uploads/a", timedelta(minutes=5))

    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
        "http://rustfs:9000/bucket/uploads/a"
    )
    assert parse_qs(parts.query)["X-Amz-Expires"] == ["300"]


async def test_put_and_get_signatures_differ(storage: ObjectStorage) -> None:
    expires_in = timedelta(minutes=5)

    put = await storage.sign_put("uploads/a", expires_in)
    get = await storage.sign_get("uploads/a", expires_in)

    assert put != get


async def test_delete_removes_the_object(
    storage: ObjectStorage, store: MemoryStore
) -> None:
    await put_async(store, "uploads/a", b"audio")
    await put_async(store, "uploads/b", b"audio")

    await storage.delete("uploads/a")

    objects = await list_objects(store).collect_async()
    assert [o["path"] for o in objects] == ["uploads/b"]


async def test_delete_of_missing_object_is_not_an_error(
    storage: ObjectStorage,
) -> None:
    await storage.delete("uploads/missing")
