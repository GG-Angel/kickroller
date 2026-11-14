import sys

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)


def configure_logging() -> None:
    logger.remove()
    logger.add(sink=sys.stdout, format=LOG_FORMAT)
    logger.enable(name="api")
