"""Fixtures shared across the kickroller tests."""

from collections.abc import Iterator

import pytest
from loguru import logger


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Loguru output, as '<function> <level> <message>' lines."""
    logs: list[str] = []
    sink = logger.add(logs.append, format="{function} {level} {message}")
    yield logs
    logger.remove(sink)
