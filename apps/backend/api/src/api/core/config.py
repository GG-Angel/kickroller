from typing import Annotated

from pydantic import Field, PostgresDsn, SecretStr
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

    endpoint: str = "http://localhost:9000"
    """S3 endpoint. Defaults to the local compose RustFS."""
    bucket: str = "kickroller"
    access_key_id: str = Field(init=False)
    secret_access_key: SecretStr = Field(init=False)
    region: str = "us-east-1"


class ServerConfig(BaseSettings):
    """HTTP server settings, read from `SERVER_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    """Interface to bind. Use 0.0.0.0 inside a container."""
    port: Port = 8000
    metrics_port: Port = 9100
    """Prometheus metrics port. Keep it off the public network."""


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
