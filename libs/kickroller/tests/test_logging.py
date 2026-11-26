import logging
from collections.abc import Iterator

import pytest

from kickroller.logging import InterceptHandler


@pytest.fixture
def std_logger() -> Iterator[logging.Logger]:
    """A stdlib logger whose records go only to InterceptHandler."""
    std = logging.getLogger("tests.intercept")
    handler = InterceptHandler()
    std.addHandler(handler)
    std.propagate = False
    std.setLevel(logging.DEBUG)
    yield std
    std.removeHandler(handler)


def test_forwards_records_from_the_real_caller(
    logs: list[str], std_logger: logging.Logger
) -> None:
    std_logger.warning("hello %s", "world")

    assert logs == [
        "test_forwards_records_from_the_real_caller WARNING hello world\n"
    ]


def test_forwards_custom_levels_by_number(
    logs: list[str], std_logger: logging.Logger
) -> None:
    std_logger.log(15, "between debug and info")

    assert [m.split(" ", 1)[1] for m in logs] == [
        "Level 15 between debug and info\n"
    ]
