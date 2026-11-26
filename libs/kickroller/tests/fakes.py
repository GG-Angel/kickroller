"""Test helpers shared across the kickroller tests."""

from typing import Any

from pydantic_settings import BaseSettings


def load_settings[T: BaseSettings](cls: type[T], **fields: Any) -> T:
    """Settings from `fields` without the developer's .env file."""
    # shortcut: exported shell env vars still apply, isolate them if CI sets any.
    return cls(_env_file=None, **fields)
