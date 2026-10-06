import sys

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    if verbose and quiet:
        raise ValueError("Cannot be both verbose and quiet.")

    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"

    logger.remove()
    logger.add(sink=sys.stderr, level=level, format=LOG_FORMAT)
    logger.enable(name="trainer")
