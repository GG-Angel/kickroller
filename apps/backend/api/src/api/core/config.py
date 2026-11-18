from datetime import timedelta
from ipaddress import IPv4Address, IPv6Address
from typing import Annotated

from pydantic import AmqpDsn, Field, HttpUrl, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Port = Annotated[int, Field(ge=1, le=65535)]


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
    public_endpoint: HttpUrl | None = None
    """Endpoint clients reach for signed URLs. Defaults to `endpoint`."""
    bucket: str = "kickroller"
    access_key_id: str = Field(init=False)
    secret_access_key: SecretStr = Field(init=False)
    region: str = "us-east-1"

    upload_url_ttl: timedelta = Field(
        default=timedelta(minutes=15), gt=timedelta(0)
    )
    """How long a signed upload URL and its pending upload stay valid."""
    playback_url_ttl: timedelta = Field(
        default=timedelta(hours=1), gt=timedelta(0)
    )
    """How long a signed playback URL stays valid. Seeking needs it unexpired."""


class BrokerConfig(BaseSettings):
    """RabbitMQ connection settings, read from `BROKER_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="BROKER_",
        env_file=".env",
        extra="ignore",
    )

    url: AmqpDsn = Field(init=False)
    """AMQP URI with credentials, e.g. amqp://user:pass@localhost:5672/."""


class ServerConfig(BaseSettings):
    """HTTP server settings, read from `SERVER_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        env_file=".env",
        extra="ignore",
    )

    host: IPv4Address | IPv6Address = IPv4Address("127.0.0.1")
    """Interface to bind. Use 0.0.0.0 inside a container."""
    port: Port = 8000
    """Port for the public API."""
    metrics_port: Port = 9100
    """Prometheus metrics port. Keep it off the public network."""


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
