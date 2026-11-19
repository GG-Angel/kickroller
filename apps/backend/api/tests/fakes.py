"""Test data shared across the API tests."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic_settings import BaseSettings

from api.models import Track, TrackStatus

USER_ID = UUID("01900000-0000-7000-8000-0000000000aa")
OTHER_USER_ID = UUID("01900000-0000-7000-8000-0000000000bb")
TRACK_ID = UUID("01900000-0000-7000-8000-000000000001")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def load_settings[T: BaseSettings](cls: type[T], **fields: Any) -> T:
    """Settings from `fields` without the developer's .env file."""
    # shortcut: exported shell env vars still apply, isolate them if CI sets any.
    return cls(_env_file=None, **fields)


def make_track(**overrides: Any) -> Track:
    """A ready track owned by USER_ID, with any field overridden."""
    fields: dict[str, Any] = {
        "id": TRACK_ID,
        "user_id": USER_ID,
        "title": "Kick",
        "artist": "DJ",
        "object_key": f"tracks/{TRACK_ID}",
        "duration_ms": 1000,
        "status": TrackStatus.READY,
        "created_at": NOW,
        "updated_at": NOW,
        "removed_at": None,
    }
    return Track(**(fields | overrides))
