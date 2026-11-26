"""Validation rules of the settings."""

from ipaddress import IPv4Address, IPv6Address

import pytest
from pydantic import ValidationError

from api.core.config import (
    ServerConfig,
    StorageConfig,
)
from tests.fakes import load_settings

STORAGE_KEYS = {"access_key_id": "key", "secret_access_key": "secret"}


@pytest.mark.parametrize(
    ("host", "parsed"),
    [("0.0.0.0", IPv4Address("0.0.0.0")), ("::", IPv6Address("::"))],
)
def test_server_host_accepts_ipv4_and_ipv6(host: str, parsed: object) -> None:
    assert load_settings(ServerConfig, host=host).host == parsed


def test_server_docs_are_off_by_default() -> None:
    assert load_settings(ServerConfig).docs is False


@pytest.mark.parametrize(
    "fields", [{"host": "localhost"}, {"port": 0}, {"metrics_port": 65536}]
)
def test_server_rejects_invalid_bind(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_settings(ServerConfig, **fields)


@pytest.mark.parametrize("ttl", ["upload_url_ttl", "playback_url_ttl"])
def test_storage_ttls_must_be_positive(ttl: str) -> None:
    with pytest.raises(ValidationError):
        load_settings(StorageConfig, **STORAGE_KEYS, **{ttl: 0})


@pytest.mark.parametrize(
    ("public_endpoint", "endpoint"),
    [(None, "http://localhost:9000/"), ("http://cdn.test", "http://cdn.test/")],
    ids=["internal", "public"],
)
def test_storage_signs_against_the_client_endpoint(
    public_endpoint: str | None, endpoint: str
) -> None:
    config = load_settings(
        StorageConfig, **STORAGE_KEYS, public_endpoint=public_endpoint
    )

    assert str(config.for_signing().endpoint) == endpoint
