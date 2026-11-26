from datetime import UTC, datetime
from uuid import UUID

from fastapi import Depends
from loguru import logger
from obstore.exceptions import BaseError

from api.mappers.track import to_track_response
from api.models import Track, TrackStatus
from api.repositories.track import TrackRepository, get_track_repository
from api.schemas import (
    PlaybackUrlResponse,
    TrackListResponse,
    TrackResponse,
    UpdateTrackRequest,
)
from api.services.storage import StorageService, get_storage_service


class TrackService:
    """Validated tracks."""

    def __init__(
        self, tracks: TrackRepository, storage: StorageService
    ) -> None:
        self._tracks = tracks
        self._storage = storage

    async def list_ready_by_owner(
        self,
        owner_id: UUID,
        limit: int,
        cursor: UUID | None,
    ) -> TrackListResponse:
        """A page of the owner's ready tracks, newest first. Cursor: last id."""
        # One extra row tells us whether another page exists.
        tracks_with_lookahead = await self._tracks.list_ready_by_owner(
            owner_id, limit + 1, cursor
        )
        tracks_on_page = tracks_with_lookahead[:limit]
        has_next_page = len(tracks_with_lookahead) > limit
        return TrackListResponse(
            items=[to_track_response(track) for track in tracks_on_page],
            next_cursor=str(tracks_on_page[-1].id) if has_next_page else None,
        )

    async def get_owned_ready_track(
        self, user_id: UUID, track_id: UUID
    ) -> Track | None:
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
        await self._tracks.commit()
        return to_track_response(track)

    async def remove(self, track: Track) -> None:
        """Mark the track removed, then delete its audio."""
        track.status = TrackStatus.REMOVED
        track.removed_at = datetime.now(UTC)
        await self._tracks.commit()
        try:
            await self._storage.delete(track.object_key)
        except BaseError:
            logger.exception(
                f"Failed to delete audio of removed track {track.id}"
            )

    async def create_playback_url(self, track: Track) -> PlaybackUrlResponse:
        """Sign a GET URL for the track's audio."""
        playback_url_ttl = self._storage.playback_url_ttl
        expires_at = datetime.now(UTC) + playback_url_ttl
        playback_url = await self._storage.sign_get(
            key=track.object_key, expires_in=playback_url_ttl
        )
        return PlaybackUrlResponse(url=playback_url, expires_at=expires_at)


def get_track_service(
    tracks: TrackRepository = Depends(get_track_repository),
    storage: StorageService = Depends(get_storage_service),
) -> TrackService:
    """FastAPI dependency that returns a track service for one request."""
    return TrackService(tracks, storage)
