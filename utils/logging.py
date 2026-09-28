from __future__ import annotations

import logging
from pathlib import Path

from loguru import logger


def configure_logging() -> None:
    log_path = Path(__file__).resolve().parent.parent / "app.log"
    logger.remove()
    logger.add(log_path, rotation="10 MB", retention="10 days")
    logger.add(logging.StreamHandler(), format="{time} {level} {message}")
