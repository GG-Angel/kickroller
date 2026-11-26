"""Validation rules of the settings."""

import pytest
from pydantic import ValidationError

from ingester.core.config import BrokerConfig, StorageConfig
from tests.fakes import load_settings


def test_storage_requires_credentials() -> None:
    with pytest.raises(ValidationError):
        load_settings(StorageConfig)


def test_broker_url_must_be_amqp() -> None:
    url = "amqp://u:p@rabbitmq:5672/"
    assert str(load_settings(BrokerConfig, url=url).url) == url
    with pytest.raises(ValidationError):
        load_settings(BrokerConfig, url="http://rabbitmq:5672/")
