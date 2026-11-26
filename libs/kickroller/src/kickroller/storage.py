from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from aiobotocore.config import AioConfig
from aiobotocore.session import get_session
from loguru import logger
from types_aiobotocore_s3 import S3Client

from kickroller.config import StorageConfig


@asynccontextmanager
async def open_store(
    config: StorageConfig, *, check_bucket_access: bool = True
) -> AsyncGenerator[S3Client]:
    """Open an S3 client and close it on exit. Checks the bucket if asked."""
    async with get_session().create_client(
        "s3",
        endpoint_url=str(config.endpoint),
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key.get_secret_value(),
        region_name=config.region,
        config=AioConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    ) as store:
        if check_bucket_access:
            await store.head_bucket(Bucket=config.bucket)
            logger.info(
                f"Connected to bucket {config.bucket} at {config.endpoint}"
            )
        yield store
