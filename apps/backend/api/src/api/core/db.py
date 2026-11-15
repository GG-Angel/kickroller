from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import Depends
from loguru import logger

from api.core.config import DatabaseConfig
from api.core.state import AppState, get_app_state


@asynccontextmanager
async def open_db_pool(
    config: DatabaseConfig,
) -> AsyncGenerator[asyncpg.Pool]:
    """Open a connection pool, check that the database answers, close on exit."""
    async with asyncpg.create_pool(
        dsn=str(config.url),
        min_size=config.pool_min_size,
        max_size=config.pool_max_size,
        command_timeout=config.command_timeout,
    ) as pool:
        await pool.execute("SELECT 1")
        logger.info(
            f"Connected to Postgres at {config.url.hosts()[0]['host']} "
            f"(pool {config.pool_min_size}-{config.pool_max_size})"
        )
        yield pool
    logger.info("Closed Postgres pool")


async def get_connection(
    state: AppState = Depends(get_app_state),
) -> AsyncGenerator[asyncpg.pool.PoolConnectionProxy]:
    """FastAPI dependency that borrows a connection for one request."""
    async with state.db_pool.acquire() as connection:
        yield connection
