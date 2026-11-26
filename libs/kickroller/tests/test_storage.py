"""Unit tests for building the store. Nothing here connects to RustFS."""

import pytest
from pydantic import HttpUrl

from kickroller.config import StorageConfig
from kickroller.storage import build_store
from tests.fakes import load_settings


@pytest.mark.parametrize(
    ("endpoint", "allow_http"),
    [("http://rustfs:9000", "true"), ("https://s3.example.com", "false")],
)
def test_build_store_allows_http_only_for_http_endpoints(
    endpoint: str, allow_http: str
) -> None:
    config = load_settings(
        StorageConfig, access_key_id="key", secret_access_key="secret"
    )

    store = build_store(config, HttpUrl(endpoint))

    assert store.client_options == {"allow_http": allow_http}
    assert store.config == {
        "endpoint": f"{endpoint}/",
        "bucket": "kickroller",
        "region": "us-east-1",
        "access_key_id": "key",
        "secret_access_key": "secret",
    }
