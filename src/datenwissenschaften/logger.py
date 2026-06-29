import sys
from pathlib import Path

from loguru import logger

from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config


def _runtime_context() -> str:
    try:
        from datenwissenschaften.runtime import get_runtime

        runtime = get_runtime()
    except Exception:
        return "[game=- state=-]"

    return f"[game={runtime.game} state={runtime.savestate}]"


def _add_runtime_context(record: dict) -> None:
    record["extra"]["runtime_context"] = _runtime_context()


def setup_logging(level: str | None = None, *, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    if level is None:
        level = load_config(config_path).log_level

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
        level=level,
    )
