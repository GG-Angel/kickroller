from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Postgres connection settings, read from `DATABASE_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        extra="ignore",
    )

    url: PostgresDsn = Field(init=False)
    pool_min_size: int = Field(default=1, ge=0)
    """Connections the pool keeps open."""
    pool_max_size: int = Field(default=10, ge=1)
    """Most connections the pool opens at once."""
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
    port: int = Field(default=8000, ge=1, le=65535)


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


CONFIG = Config()
