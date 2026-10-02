import sys

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Log to standard error, so that standard output only has the result."""
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"
    logger.remove()
    logger.add(sink=sys.stderr, level=level, format=LOG_FORMAT)
    logger.enable(name="analyzer")
