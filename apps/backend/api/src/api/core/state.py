from dataclasses import dataclass

from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import AsyncEngine
from types_aiobotocore_s3 import S3Client

from api.core.config import Config


@dataclass(frozen=True, slots=True)
class AppState:
    """Resources that live for the whole app, created in the lifespan."""

    config: Config
    db_engine: AsyncEngine
    store: S3Client
    url_signing_store: S3Client
    broker: RabbitBroker
