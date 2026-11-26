from urllib.parse import unquote_plus

from pydantic import BaseModel, Field


class EventObject(BaseModel):
    key: str
    """URL-encoded object key."""


class EventS3(BaseModel):
    object: EventObject


class EventRecord(BaseModel):
    s3: EventS3


class StorageEvent(BaseModel):
    records: list[EventRecord] = Field(alias="Records")

    def decode_object_keys(self) -> list[str]:
        return [unquote_plus(record.s3.object.key) for record in self.records]
