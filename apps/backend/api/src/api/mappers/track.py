from api.models import Track
from api.schemas import TrackResponse


def to_track_response(track: Track) -> TrackResponse:
    """Map a track row to its API response."""
    return TrackResponse(
        id=track.id,
        title=track.title,
        artist=track.artist,
        duration_ms=track.duration_ms,
        status=track.status,
        created_at=track.created_at,
        updated_at=track.updated_at,
        removed_at=track.removed_at,
    )
