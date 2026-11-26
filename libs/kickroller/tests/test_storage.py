"""Unit tests for opening the store. Nothing here connects to RustFS."""

import pytest

from kickroller.config import StorageConfig
from kickroller.storage import open_store
from tests.fakes import load_settings

pytestmark = pytest.mark.anyio


async def test_store_uses_the_configured_endpoint_and_region() -> None:
    config = load_settings(
        StorageConfig,
        endpoint="http://rustfs:9000",
        access_key_id="key",
        secret_access_key="secret",
    )

    async with open_store(config, check_bucket_access=False) as store:
        assert store.meta.endpoint_url == "http://rustfs:9000/"
        assert store.meta.region_name == "us-east-1"
