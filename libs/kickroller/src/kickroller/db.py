from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

from kickroller.config import DatabaseConfig


@asynccontextmanager
async def open_db(config: DatabaseConfig) -> AsyncGenerator[AsyncEngine]:
    """Open an engine, check that the database answers, dispose on exit."""
    engine = create_async_engine(
        str(config.url),
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        connect_args={"command_timeout": config.command_timeout},
    )
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))  # Test the connection
        logger.info(
            f"Connected to Postgres at {config.url.hosts()[0]['host']} "
            f"(pool {config.pool_size}+{config.max_overflow})"
        )
        yield engine
    finally:
        await engine.dispose()
        logger.info("Closed Postgres engine")
