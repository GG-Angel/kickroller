"""Unit tests for TrackRepository. The session is mocked, so no Postgres."""

from unittest.mock import Mock

import pytest
from sqlalchemy import ScalarResult, Select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import Track, TrackStatus
from api.repositories.track import TrackRepository
from tests.fakes import TRACK_ID, USER_ID, make_track

pytestmark = pytest.mark.anyio


@pytest.fixture
def session() -> Mock:
    session = Mock(spec=AsyncSession)
    session.scalars.return_value = Mock(spec=ScalarResult)
    return session


def compiled_query(session: Mock) -> tuple[str, dict[str, object]]:
    """The SQL and params of the query passed to session.scalars."""
    query: Select = session.scalars.call_args.args[0]
    compiled = query.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


async def test_list_without_filters_orders_newest_first(
    session: Mock,
) -> None:
    rows = [make_track()]
    session.scalars.return_value.all.return_value = rows

    assert await TrackRepository(session).list(None, None, 20, None) == rows

    sql, params = compiled_query(session)
    assert "WHERE" not in sql
    assert "ORDER BY track.id DESC" in sql
    assert params == {"param_1": 20}


async def test_list_applies_owner_status_and_cursor(session: Mock) -> None:
    await TrackRepository(session).list(USER_ID, TrackStatus.READY, 5, TRACK_ID)

    sql, params = compiled_query(session)
    assert "track.user_id = " in sql
    assert "track.status = " in sql
    assert "track.id < " in sql
    assert params == {
        "user_id_1": USER_ID,
        "status_1": TrackStatus.READY,
        "id_1": TRACK_ID,
        "param_1": 5,
    }


async def test_get_loads_by_id(session: Mock) -> None:
    track = make_track()
    session.get.return_value = track

    assert await TrackRepository(session).get(TRACK_ID) is track
    session.get.assert_awaited_once_with(Track, TRACK_ID)


async def test_save_commits(session: Mock) -> None:
    await TrackRepository(session).save()

    session.commit.assert_awaited_once()
