"""
Centralised logger using loguru.
Import `logger` from here everywhere — don't use print() in production code.
"""

import sys
from loguru import logger
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler, set our format
logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}",
    level="INFO",
    colorize=True,
)

logger.add(
    LOG_DIR / "app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
)

__all__ = ["logger"]
