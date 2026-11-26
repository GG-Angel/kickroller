from loguru import logger
from obstore.store import S3Store

from kickroller.config import StorageConfig


async def create_store(
    config: StorageConfig, *, check_bucket_access: bool = True
) -> S3Store:
    """Build the bucket's S3 store. Lists the bucket to fail fast if asked."""
    store = S3Store(
        config.bucket,
        endpoint=str(config.endpoint),
        access_key_id=config.access_key_id,
        secret_access_key=config.secret_access_key.get_secret_value(),
        region=config.region,
        client_options={"allow_http": config.endpoint.scheme == "http"},
    )
    if check_bucket_access:
        await store.list_with_delimiter_async()
        logger.info(f"Connected to bucket {config.bucket} at {config.endpoint}")
    return store
