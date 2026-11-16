from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends
from loguru import logger
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)

from api.core.config import DatabaseConfig
from api.core.state import AppState, get_app_state


@asynccontextmanager
async def open_db(config: DatabaseConfig) -> AsyncGenerator[AsyncEngine]:
    """Open an engine, check that the database answers, dispose on exit."""
    engine = create_async_engine(
        make_url(str(config.url)).set(drivername="postgresql+asyncpg"),
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


async def get_session(
    state: AppState = Depends(get_app_state),
) -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that opens a session for one request."""
    async with AsyncSession(state.db, expire_on_commit=False) as session:
        yield session
