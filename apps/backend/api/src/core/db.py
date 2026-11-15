from collections.abc import AsyncGenerator

import asyncpg
from fastapi import Depends
from loguru import logger

from src.core.config import DatabaseConfig
from src.core.state import AppState, get_app_state


async def create_pool(config: DatabaseConfig) -> asyncpg.Pool:
    """Open a connection pool and check that the database answers."""
    pool = await asyncpg.create_pool(
        dsn=str(config.url),
        min_size=config.pool_min_size,
        max_size=config.pool_max_size,
        command_timeout=config.command_timeout,
    )
    await pool.execute("SELECT 1")
    logger.info(
        f"Connected to Postgres at {config.url.hosts()[0]['host']} "
        f"(pool {config.pool_min_size}-{config.pool_max_size})"
    )
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()
    logger.info("Closed Postgres pool")


async def get_connection(
    state: AppState = Depends(get_app_state),
) -> AsyncGenerator[asyncpg.pool.PoolConnectionProxy]:
    """FastAPI dependency that borrows a connection for one request."""
    async with state.db_pool.acquire() as connection:
        yield connection
