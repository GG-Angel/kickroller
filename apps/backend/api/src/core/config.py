from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Postgres connection settings, read from `DATABASE_*` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_", env_file=".env", extra="ignore"
    )

    url: PostgresDsn = PostgresDsn(
        "postgresql://kickroller:mysecretpassword@localhost:5432/kickroller"
    )
    """Connection string. Defaults to the local compose database."""
    pool_min_size: int = Field(default=1, ge=0)
    """Connections the pool keeps open."""
    pool_max_size: int = Field(default=10, ge=1)
    """Most connections the pool opens at once."""
    command_timeout: float | None = 30.0
    """Default query timeout, in seconds. None disables it."""


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


CONFIG = Config()
