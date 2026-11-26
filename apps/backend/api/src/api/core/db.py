from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.state import AppState, get_app_state


async def get_session(
    state: AppState = Depends(get_app_state),
) -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that opens a session for one request."""
    async with AsyncSession(state.db_engine, expire_on_commit=False) as session:
        yield session
