"""HTTP contract of the track routes. The track service is mocked."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.mappers.track import to_track_response
from api.schemas import PlaybackUrlResponse, TrackListResponse
from api.services.track import TrackService, get_track_service
from tests.fakes import NOW, OTHER_USER_ID, TRACK_ID, USER_ID, make_track

TRACKS_URL = "/me/tracks"
TRACK_URL = f"{TRACKS_URL}/{TRACK_ID}"


@pytest.fixture
def tracks(app: FastAPI) -> Mock:
    tracks = Mock(spec=TrackService)
    tracks.get_owned.return_value = make_track()
    app.dependency_overrides[get_track_service] = lambda: tracks
    return tracks


def test_get_track_returns_own_track(client: TestClient, tracks: Mock) -> None:
    response = client.get(TRACK_URL)

    assert response.status_code == 200
    assert response.json() == to_track_response(make_track()).model_dump(
        mode="json"
    )
    tracks.get_owned.assert_awaited_once_with(USER_ID, TRACK_ID)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", TRACK_URL),
        ("PATCH", TRACK_URL),
        ("DELETE", TRACK_URL),
        ("POST", f"{TRACK_URL}/playback-url"),
    ],
)
def test_track_routes_404_when_not_own_track(
    client: TestClient, tracks: Mock, method: str, path: str
) -> None:
    tracks.get_owned.return_value = None

    response = client.request(method, path, json={"title": "New"})

    assert response.status_code == 404
    tracks.update.assert_not_called()
    tracks.delete.assert_not_called()
    tracks.create_playback_url.assert_not_called()


def test_track_routes_require_a_user_id(
    client: TestClient, tracks: Mock
) -> None:
    del client.headers["X-User-Id"]

    assert client.get(TRACK_URL).status_code == 422
    tracks.get_owned.assert_not_called()


def test_list_tracks_passes_filters_and_cursor(
    client: TestClient, tracks: Mock
) -> None:
    page = TrackListResponse(items=[], next_cursor=None)
    tracks.list.return_value = page

    response = client.get(
        TRACKS_URL,
        params={"limit": 5, "cursor": str(TRACK_ID)},
    )

    assert response.status_code == 200
    assert response.json() == page.model_dump(mode="json")
    tracks.list.assert_awaited_once_with(USER_ID, 5, TRACK_ID)


def test_list_tracks_defaults_to_twenty(
    client: TestClient, tracks: Mock
) -> None:
    tracks.list.return_value = TrackListResponse(items=[], next_cursor=None)

    assert client.get(TRACKS_URL).status_code == 200
    tracks.list.assert_awaited_once_with(USER_ID, 20, None)


def test_list_tracks_lists_only_the_callers_tracks(
    client: TestClient, tracks: Mock
) -> None:
    tracks.list.return_value = TrackListResponse(items=[], next_cursor=None)

    # An owner_id from the client is ignored, not trusted.
    client.get(TRACKS_URL, params={"owner_id": str(OTHER_USER_ID)})

    assert tracks.list.call_args.args[0] == USER_ID


def test_list_tracks_requires_a_user_id(
    client: TestClient, tracks: Mock
) -> None:
    del client.headers["X-User-Id"]

    assert client.get(TRACKS_URL).status_code == 422
    tracks.list.assert_not_called()


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"cursor": "x"}],
    ids=["limit-low", "limit-high", "bad-cursor"],
)
def test_list_tracks_rejects_invalid_params(
    client: TestClient, tracks: Mock, params: dict[str, str | int]
) -> None:
    assert client.get(TRACKS_URL, params=params).status_code == 422
    tracks.list.assert_not_called()


def test_update_track_applies_changes_to_own_track(
    client: TestClient, tracks: Mock
) -> None:
    updated = to_track_response(make_track(title="New"))
    tracks.update.return_value = updated

    response = client.patch(TRACK_URL, json={"title": "New"})

    assert response.status_code == 200
    assert response.json() == updated.model_dump(mode="json")
    track, changes = tracks.update.call_args.args
    assert track.id == TRACK_ID
    assert changes.title == "New"


def test_update_track_rejects_null_title(
    client: TestClient, tracks: Mock
) -> None:
    assert client.patch(TRACK_URL, json={"title": None}).status_code == 422
    tracks.update.assert_not_called()


def test_delete_track_removes_own_track(
    client: TestClient, tracks: Mock
) -> None:
    response = client.delete(TRACK_URL)

    assert response.status_code == 204
    assert tracks.delete.call_args.args[0].id == TRACK_ID


def test_create_playback_url_signs_own_track(
    client: TestClient, tracks: Mock
) -> None:
    signed = PlaybackUrlResponse(url="http://rustfs/get", expires_at=NOW)
    tracks.create_playback_url.return_value = signed

    response = client.post(f"{TRACK_URL}/playback-url")

    assert response.status_code == 200
    assert response.json() == signed.model_dump(mode="json")
    assert tracks.create_playback_url.call_args.args[0].id == TRACK_ID
