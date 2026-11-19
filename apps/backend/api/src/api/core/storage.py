from loguru import logger
from obstore.store import ObjectStore, S3Store
from pydantic import HttpUrl

from api.core.config import StorageConfig
from api.services.storage import ObjectStorage


def build_store(config: StorageConfig, endpoint: HttpUrl) -> S3Store:
    """Build an S3 store for the bucket at `endpoint`. Does not connect."""
    return S3Store(
        config.bucket,
        endpoint=str(endpoint),
        access_key_id=config.access_key_id,
        secret_access_key=config.secret_access_key.get_secret_value(),
        region=config.region,
        client_options={"allow_http": endpoint.scheme == "http"},
    )


async def create_store(config: StorageConfig) -> S3Store:
    """Build the S3 store and check that the bucket answers."""
    store = build_store(config, config.endpoint)
    await store.list_with_delimiter_async()
    logger.info(f"Connected to bucket {config.bucket} at {config.endpoint}")
    return store


def build_object_storage(
    config: StorageConfig, store: ObjectStore
) -> ObjectStorage:
    """Wrap `store`, signing URLs against the endpoint clients can reach."""
    signer = build_store(config, config.public_endpoint or config.endpoint)
    return ObjectStorage(
        store,
        signer,
        config.upload_url_ttl,
        config.playback_url_ttl,
    )
