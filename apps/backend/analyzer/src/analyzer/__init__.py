from loguru import logger

from analyzer.detect import (
    Detection,
    DetectorConfig,
    detect_kicks,
    detect_kicks_in_signal,
)

# Silent when used as a library; call logger.enable("analyzer") to see the logs.
logger.disable("analyzer")

__all__ = ["Detection", "DetectorConfig", "detect_kicks", "detect_kicks_in_signal"]
