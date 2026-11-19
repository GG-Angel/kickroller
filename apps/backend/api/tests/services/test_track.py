"""Unit tests for TrackService. The repository and storage are mocked."""

from datetime import UTC, datetime
from unittest.mock import Mock, call
from uuid import uuid7

import pytest
from obstore.exceptions import BaseError

from api.models import TrackStatus
from api.repositories.track import TrackRepository
from api.schemas import UpdateTrackRequest
from api.services.track import TrackService
from tests.fakes import OTHER_USER_ID, TRACK_ID, USER_ID, make_track

pytestmark = pytest.mark.anyio


@pytest.fixture
def repo() -> Mock:
    return Mock(spec=TrackRepository)


@pytest.fixture
def service(repo: Mock, storage: Mock) -> TrackService:
    return TrackService(repo, storage)


async def test_list_fetches_one_extra_row_to_find_the_next_page(
    service: TrackService, repo: Mock
) -> None:
    rows = [make_track(id=uuid7()) for _ in range(3)]
    repo.list.return_value = rows

    page = await service.list(USER_ID, 2, TRACK_ID)

    repo.list.assert_awaited_once_with(USER_ID, 3, TRACK_ID)
    assert [item.id for item in page.items] == [rows[0].id, rows[1].id]
    assert page.next_cursor == str(rows[1].id)


async def test_list_last_page_has_no_cursor(
    service: TrackService, repo: Mock
) -> None:
    rows = [make_track(id=uuid7()) for _ in range(2)]
    repo.list.return_value = rows

    page = await service.list(USER_ID, 2, None)

    assert [item.id for item in page.items] == [row.id for row in rows]
    assert page.next_cursor is None


async def test_list_empty(service: TrackService, repo: Mock) -> None:
    repo.list.return_value = []

    page = await service.list(USER_ID, 20, None)

    assert page.items == []
    assert page.next_cursor is None


async def test_get_owned_returns_own_ready_track(
    service: TrackService, repo: Mock
) -> None:
    track = make_track()
    repo.get.return_value = track

    assert await service.get_owned(USER_ID, TRACK_ID) is track
    repo.get.assert_awaited_once_with(TRACK_ID)


@pytest.mark.parametrize(
    "track",
    [
        None,
        make_track(user_id=OTHER_USER_ID),
        make_track(status=TrackStatus.REMOVED),
    ],
    ids=["missing", "other-user", "removed"],
)
async def test_get_owned_hides_tracks_the_user_cannot_see(
    service: TrackService, repo: Mock, track: object
) -> None:
    repo.get.return_value = track

    assert await service.get_owned(USER_ID, TRACK_ID) is None


async def test_update_sets_title_and_keeps_unsent_artist(
    service: TrackService, repo: Mock
) -> None:
    track = make_track()

    response = await service.update(track, UpdateTrackRequest(title="New"))

    assert (track.title, track.artist) == ("New", "DJ")
    assert (response.title, response.artist) == ("New", "DJ")
    repo.save.assert_awaited_once()


async def test_update_clears_artist_sent_as_null(
    service: TrackService, repo: Mock
) -> None:
    track = make_track()

    await service.update(track, UpdateTrackRequest(artist=None))

    assert (track.title, track.artist) == ("Kick", None)
    repo.save.assert_awaited_once()


async def test_delete_removes_track_then_deletes_audio(
    service: TrackService, repo: Mock, storage: Mock
) -> None:
    track = make_track()
    calls = Mock()
    calls.attach_mock(repo.save, "save")
    calls.attach_mock(storage.delete, "delete")

    await service.delete(track)

    assert track.status == TrackStatus.REMOVED
    assert track.removed_at is not None
    # Save first, so a failed delete never leaves the track visible.
    assert calls.mock_calls == [call.save(), call.delete(track.object_key)]


async def test_delete_keeps_track_removed_when_audio_delete_fails(
    service: TrackService, repo: Mock, storage: Mock
) -> None:
    track = make_track()
    storage.delete.side_effect = BaseError("rustfs down")

    await service.delete(track)

    assert track.status == TrackStatus.REMOVED
    repo.save.assert_awaited_once()


async def test_create_playback_url_signs_get_for_the_ttl(
    service: TrackService, storage: Mock
) -> None:
    track = make_track()
    before = datetime.now(UTC)

    response = await service.create_playback_url(track)

    assert response.url == "http://rustfs/get"
    ttl = storage.playback_url_ttl
    assert before + ttl <= response.expires_at <= datetime.now(UTC) + ttl
    storage.sign_get.assert_awaited_once_with(
        key=track.object_key, expires_in=ttl
    )
