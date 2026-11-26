from pydantic import AmqpDsn, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseSettings):
    """Object storage settings, read from `STORAGE_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="STORAGE_",
        env_file=".env",
        extra="ignore",
    )

    endpoint: HttpUrl = HttpUrl("http://localhost:9000")
    """S3 endpoint. Defaults to the local compose RustFS."""
    bucket: str = "kickroller"
    access_key_id: str = Field(init=False)
    secret_access_key: SecretStr = Field(init=False)
    region: str = "us-east-1"


class BrokerConfig(BaseSettings):
    """RabbitMQ connection settings, read from `BROKER_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="BROKER_",
        env_file=".env",
        extra="ignore",
    )

    url: AmqpDsn = Field(init=False)
    """AMQP URI with credentials, e.g. amqp://user:pass@localhost:5672/."""


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    storage: StorageConfig = Field(default_factory=StorageConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
