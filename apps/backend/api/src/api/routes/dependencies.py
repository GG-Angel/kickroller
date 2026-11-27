"""FastAPI dependencies. The layers below routes never import FastAPI."""

from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status
from faststream.rabbit import RabbitBroker
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.state import AppState
from api.db.track_repository import TrackDbRepository
from api.db.upload_repository import UploadDbRepository
from api.services.storage_service import StorageService
from api.services.track_service import TrackService
from api.services.upload_service import UploadService
from kickroller.models import Track


def get_app_state(request: Request) -> AppState:
    """FastAPI dependency that returns the typed app state."""
    return request.app.state.app_state


def get_user_id(x_user_id: Annotated[UUID, Header()]) -> UUID:
    """FastAPI dependency that returns the caller's user id."""
    # shortcut: trusts the X-User-Id header, verify the JWT subject before exposing the API.
    return x_user_id


async def get_session(
    state: AppState = Depends(get_app_state),
) -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that opens a session for one request."""
    async with AsyncSession(state.db_engine, expire_on_commit=False) as session:
        yield session


def get_broker(state: AppState = Depends(get_app_state)) -> RabbitBroker:
    """FastAPI dependency that returns the shared broker for publishing."""
    return state.broker


def get_storage_service(
    state: AppState = Depends(get_app_state),
) -> StorageService:
    """FastAPI dependency that returns the storage service."""
    storage_config = state.config.storage
    return StorageService(
        state.store,
        state.url_signing_store,
        storage_config.bucket,
        storage_config.upload_url_ttl,
        storage_config.playback_url_ttl,
    )


def get_track_repository(
    session: AsyncSession = Depends(get_session),
) -> TrackDbRepository:
    """FastAPI dependency that returns a track repository for one request."""
    return TrackDbRepository(session)


def get_upload_repository(
    session: AsyncSession = Depends(get_session),
) -> UploadDbRepository:
    """FastAPI dependency that returns an upload repository for one request."""
    return UploadDbRepository(session)


def get_track_service(
    tracks: TrackDbRepository = Depends(get_track_repository),
    storage: StorageService = Depends(get_storage_service),
) -> TrackService:
    """FastAPI dependency that returns a track service for one request."""
    return TrackService(tracks, storage)


def get_upload_service(
    uploads: UploadDbRepository = Depends(get_upload_repository),
    storage: StorageService = Depends(get_storage_service),
) -> UploadService:
    """FastAPI dependency that returns an upload service for one request."""
    return UploadService(uploads, storage)


async def get_user_track(
    track_id: UUID,
    user_id: UUID = Depends(get_user_id),
    tracks: TrackService = Depends(get_track_service),
) -> Track:
    """The caller's visible track from the path, or 404."""
    track = await tracks.get_owned_ready_track(user_id, track_id)
    if track is None:  # Same 404 for others' tracks, so ids don't leak.
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return track
