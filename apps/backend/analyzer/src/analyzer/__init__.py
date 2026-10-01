from loguru import logger

from analyzer.detection.detector import Detection, detect_kicks, detect_kicks_in_signal
from analyzer.settings import DetectorSettings, Settings

# Silent when used as a library; call logger.enable("analyzer") to see the logs.
logger.disable("analyzer")

__all__ = [
    "Detection",
    "DetectorSettings",
    "Settings",
    "detect_kicks",
    "detect_kicks_in_signal",
]
