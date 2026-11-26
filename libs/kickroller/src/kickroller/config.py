from pydantic import AmqpDsn, Field, HttpUrl, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Postgres connection settings, read from `DATABASE_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        extra="ignore",
    )

    url: PostgresDsn = Field(init=False)
    """Must use the postgresql+asyncpg:// scheme."""
    pool_size: int = Field(default=5, ge=1)
    """Connections the pool keeps open."""
    max_overflow: int = Field(default=10, ge=0)
    """Extra connections the pool opens under load and closes after."""
    command_timeout: float | None = 30.0
    """Default query timeout, in seconds. None disables it."""


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
