# app/core/logger.py
# app/core/logger.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
from app.core.config import settings

# Log directory
LOG_DIR: Path = settings.BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str = __name__, log_file: Path = LOG_FILE) -> logging.Logger:
    """
    Returns a configured logger instance with colored console output and file logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        color_formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(color_formatter)

        # File handler (plain)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
