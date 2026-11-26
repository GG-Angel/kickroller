import inspect
import logging
import sys

from loguru import logger

LOG_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> | <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """Forward standard-library log records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Skip logging's own frames so loguru reports the real caller.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging_module = filename == logging.__file__
            is_importlib_bootstrap = (
                "importlib" in filename and "_bootstrap" in filename
            )
            if depth > 0 and not (is_logging_module or is_importlib_bootstrap):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging() -> None:
    logger.remove()
    logger.add(sink=sys.stdout, format=LOG_FORMAT)

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=logging.INFO,
        force=True,
    )
