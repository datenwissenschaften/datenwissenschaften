import os
import sys

from loguru import logger

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:
    find_dotenv = None
    load_dotenv = None


def _runtime_context() -> str:
    try:
        from datenwissenschaften.runtime import get_runtime

        runtime = get_runtime()
    except Exception:
        return "[game=- state=-]"

    return f"[game={runtime.game} state={runtime.savestate}]"


def _add_runtime_context(record: dict) -> None:
    record["extra"]["runtime_context"] = _runtime_context()


def setup_logging(level: str | None = None) -> None:
    if load_dotenv and find_dotenv:
        load_dotenv(find_dotenv(), override=True)

    resolved_level = level or os.environ.get("RETRO_SPEEDLAB_LOG_LEVEL", "INFO")
    logger.remove()
    logger.configure(patcher=_add_runtime_context)
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "{extra[runtime_context]} - <level>{message}</level>"
        ),
        level=resolved_level,
    )


setup_logging()
