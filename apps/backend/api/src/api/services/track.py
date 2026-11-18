from datetime import UTC, datetime
from uuid import UUID

from fastapi import Depends
from loguru import logger
from obstore.exceptions import BaseError

from api.models import Track, TrackStatus
from api.repositories.track import TrackRepository, get_track_repository
from api.schemas import (
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
)
from api.services.storage import ObjectStorage, get_object_storage


class TrackService:
    """Validated tracks."""

    def __init__(self, tracks: TrackRepository, storage: ObjectStorage) -> None:
        self._tracks = tracks
        self._storage = storage

    async def list(
        self,
        owner_id: UUID | None,
        status: TrackStatus | None,
        limit: int,
        cursor: UUID | None,
    ) -> TrackListResponse:
        """One page of tracks, newest first. The cursor is the last id seen."""
        # One extra row tells us whether another page exists.
        rows = await self._tracks.list(owner_id, status, limit + 1, cursor)
        page = rows[:limit]
        return TrackListResponse(
            items=[
                TrackResponse.model_validate(t, from_attributes=True)
                for t in page
            ],
            next_cursor=str(page[-1].id) if len(rows) > limit else None,
        )

    async def get_owned(self, user_id: UUID, track_id: UUID) -> Track | None:
        """The track if the user owns it and it isn't removed."""
        track = await self._tracks.get(track_id)
        if (
            track is None
            or track.user_id != user_id
            or track.status != TrackStatus.READY
        ):
            return None
        return track

    async def update(
        self, track: Track, changes: UpdateTrackRequest
    ) -> TrackResponse:
        """Apply changes to the track."""
        if changes.title is not None:
            track.title = changes.title
        if "artist" in changes.model_fields_set:  # Sent as null clears it.
            track.artist = changes.artist
        await self._tracks.save()
        return TrackResponse.model_validate(track, from_attributes=True)

    async def delete(self, track: Track) -> None:
        """Remove the track and delete its audio."""
        track.status = TrackStatus.REMOVED
        track.removed_at = datetime.now(UTC)
        await self._tracks.save()
        try:
            await self._storage.delete(track.object_key)
        except BaseError:
            logger.exception(
                f"Failed to delete audio of removed track {track.id}"
            )

    async def create_playback_url(self, track: Track) -> PlaybackUrlResponse:
        """Sign a GET URL for the track's audio."""
        ttl = self._storage.playback_url_ttl
        expires_at = datetime.now(UTC) + ttl
        url = await self._storage.sign_get(key=track.object_key, expires_in=ttl)
        return PlaybackUrlResponse(url=url, expires_at=expires_at)


def get_track_service(
    tracks: TrackRepository = Depends(get_track_repository),
    storage: ObjectStorage = Depends(get_object_storage),
) -> TrackService:
    """FastAPI dependency that returns a track service for one request."""
    return TrackService(tracks, storage)
