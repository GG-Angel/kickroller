from pydantic import BaseModel, Field


class EventBucket(BaseModel):
    name: str
    """Bucket name."""


class EventObject(BaseModel):
    key: str
    """URL-encoded object key."""


class EventS3(BaseModel):
    bucket: EventBucket
    object: EventObject


class EventRecord(BaseModel):
    event_name: str = Field(alias="eventName")
    s3: EventS3


class StorageEvent(BaseModel):
    """S3 event notification that RustFS publishes when an object changes."""

    event_name: str = Field(alias="EventName")
    records: list[EventRecord] = Field(alias="Records")
