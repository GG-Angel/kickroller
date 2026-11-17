from typing import Annotated
from uuid import UUID

from fastapi import Header


def get_user_id(x_user_id: Annotated[UUID, Header()]) -> UUID:
    """FastAPI dependency that returns the caller's user id."""
    # shortcut: trusts the X-User-Id header, verify the JWT subject before exposing the API.
    return x_user_id
