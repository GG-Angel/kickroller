from obstore.store import ObjectStore

from api.core.config import StorageConfig
from api.services.storage import StorageService
from kickroller.storage import build_store


def build_storage_service(
    config: StorageConfig, store: ObjectStore
) -> StorageService:
    """Wrap `store`, signing URLs against the endpoint clients can reach."""
    signer = build_store(config, config.public_endpoint or config.endpoint)
    return StorageService(
        store,
        signer,
        config.upload_url_ttl,
        config.playback_url_ttl,
    )
