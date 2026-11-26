from pathlib import Path

from kickroller.schemas import StorageEvent


def storage_event(*keys: str) -> dict[str, object]:
    records = [{"s3": {"object": {"key": key}}} for key in keys]
    return {"Records": records}


def test_object_keys_are_url_decoded() -> None:
    event = StorageEvent.model_validate(
        storage_event("uploads/a%2Fb+c%2B", "uploads/plain")
    )

    assert event.decode_object_keys() == ["uploads/a/b c+", "uploads/plain"]


def test_parses_a_real_rustfs_put_event() -> None:
    event = StorageEvent.model_validate_json(
        (Path(__file__).parent / "data" / "rustfs_put_event.json").read_text()
    )

    assert event.decode_object_keys() == [
        "uploads/01a21980-f18e-7028-a754-892fa45facf3"
    ]
