import os
import sys

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"

    # DataLoader workers read these when they import loguru
    os.environ["LOGURU_LEVEL"] = level
    os.environ["LOGURU_FORMAT"] = LOG_FORMAT

    logger.remove()
    logger.add(sink=sys.stderr, level=level, format=LOG_FORMAT)
    logger.enable(name="trainer")
