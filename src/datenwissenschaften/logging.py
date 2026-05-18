import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()
os.environ.setdefault("LOG_LEVEL", "INFO")


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.environ["LOG_LEVEL"])
