from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kickroller.config import BrokerConfig, DatabaseConfig, StorageConfig


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
