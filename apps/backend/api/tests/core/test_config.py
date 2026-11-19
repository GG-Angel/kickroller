"""Validation rules of the settings."""

from ipaddress import IPv4Address, IPv6Address

import pytest
from pydantic import ValidationError

from api.core.config import (
    BrokerConfig,
    DatabaseConfig,
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


@pytest.mark.parametrize(
    "fields", [{"host": "localhost"}, {"port": 0}, {"metrics_port": 65536}]
)
def test_server_rejects_invalid_bind(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_settings(ServerConfig, **fields)


def test_database_url_must_be_postgres() -> None:
    with pytest.raises(ValidationError):
        load_settings(DatabaseConfig, url="mysql://db/kickroller")


@pytest.mark.parametrize("ttl", ["upload_url_ttl", "playback_url_ttl"])
def test_storage_ttls_must_be_positive(ttl: str) -> None:
    with pytest.raises(ValidationError):
        load_settings(StorageConfig, **STORAGE_KEYS, **{ttl: 0})


def test_storage_requires_credentials() -> None:
    with pytest.raises(ValidationError):
        load_settings(StorageConfig)


def test_broker_url_must_be_amqp() -> None:
    url = "amqp://u:p@rabbitmq:5672/"
    assert str(load_settings(BrokerConfig, url=url).url) == url
    with pytest.raises(ValidationError):
        load_settings(BrokerConfig, url="http://rabbitmq:5672/")
