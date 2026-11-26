from datetime import timedelta
from ipaddress import IPv4Address, IPv6Address
from typing import Annotated, Self

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

import kickroller.config
from kickroller.config import BrokerConfig, DatabaseConfig

Port = Annotated[int, Field(ge=1, le=65535)]


class StorageConfig(kickroller.config.StorageConfig):
    """Object storage settings, plus what the api needs to sign URLs."""

    public_endpoint: HttpUrl | None = None
    """Endpoint clients reach for signed URLs. Defaults to `endpoint`."""
    upload_url_ttl: timedelta = Field(
        default=timedelta(minutes=15), gt=timedelta(0)
    )
    """How long a signed upload URL and its pending upload stay valid."""
    playback_url_ttl: timedelta = Field(
        default=timedelta(hours=1), gt=timedelta(0)
    )
    """How long a signed playback URL stays valid. Seeking needs it unexpired."""

    def for_signing(self) -> Self:
        if self.public_endpoint is None:
            return self
        return self.model_copy(update={"endpoint": self.public_endpoint})


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
    docs: bool = False
    """Serve the OpenAPI and AsyncAPI docs. Off so prod never exposes them."""


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
