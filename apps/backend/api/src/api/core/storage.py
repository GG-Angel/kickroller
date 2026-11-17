from loguru import logger
from obstore.store import S3Store

from api.core.config import StorageConfig


def build_store(config: StorageConfig, endpoint: str) -> S3Store:
    """Build an S3 store for the bucket at `endpoint`. Does not connect."""
    return S3Store(
        config.bucket,
        endpoint=endpoint,
        access_key_id=config.access_key_id,
        secret_access_key=config.secret_access_key.get_secret_value(),
        region=config.region,
        client_options={"allow_http": endpoint.startswith("http://")},
    )


async def create_store(config: StorageConfig) -> S3Store:
    """Build the S3 store and check that the bucket answers."""
    store = build_store(config, config.endpoint)
    await store.list_with_delimiter_async()
    logger.info(f"Connected to bucket {config.bucket} at {config.endpoint}")
    return store
