"""Unit tests for building stores. Nothing here connects to RustFS."""

from urllib.parse import urlsplit

import pytest
from obstore.store import MemoryStore

from api.core.config import StorageConfig
from api.core.storage import build_storage_service
from tests.fakes import load_settings

pytestmark = pytest.mark.anyio


def storage_config(**overrides: object) -> StorageConfig:
    return load_settings(
        StorageConfig,
        **{"access_key_id": "key", "secret_access_key": "secret"} | overrides,
    )


@pytest.mark.parametrize(
    ("public_endpoint", "signed_host"),
    [(None, "localhost:9000"), ("http://cdn.example.com", "cdn.example.com")],
    ids=["internal", "public"],
)
async def test_build_storage_service_signs_with_the_client_endpoint(
    public_endpoint: str | None, signed_host: str
) -> None:
    config = storage_config(public_endpoint=public_endpoint)

    storage = build_storage_service(config, MemoryStore())
    url = await storage.sign_get("tracks/a", config.playback_url_ttl)

    assert urlsplit(url).netloc == signed_host
    assert storage.upload_url_ttl == config.upload_url_ttl
    assert storage.playback_url_ttl == config.playback_url_ttl
