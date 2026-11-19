import pytest
from pydantic import ValidationError

from api.schemas import UpdateTrackRequest


def test_update_strips_whitespace() -> None:
    changes = UpdateTrackRequest(title="  Kick  ", artist=" DJ ")

    assert (changes.title, changes.artist) == ("Kick", "DJ")


def test_update_omitted_fields_are_unset() -> None:
    assert UpdateTrackRequest().model_fields_set == set()


def test_update_null_artist_is_set_so_it_clears() -> None:
    changes = UpdateTrackRequest(artist=None)

    assert changes.artist is None
    assert changes.model_fields_set == {"artist"}


@pytest.mark.parametrize(
    "fields",
    [{"title": None}, {"title": "   "}, {"title": "x" * 301}, {"artist": ""}],
    ids=["null-title", "blank-title", "long-title", "empty-artist"],
)
def test_update_rejects_invalid_fields(fields: dict[str, str | None]) -> None:
    with pytest.raises(ValidationError):
        UpdateTrackRequest.model_validate(fields)
