from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from datenwissenschaften.ui.telemetry import get_store


@dataclass(frozen=True)
class ModelResetRequest:
    game: str
    model_dir: Path
    on_reset: Callable[[], None] | None = None


class TrainingControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game: str | None = None
        self._model_dir: Path | None = None
        self._restart_supported = False
        self._on_reset: Callable[[], None] | None = None
        self._pending: ModelResetRequest | None = None

    def configure(
        self,
        *,
        game: str,
        model_dir: Path,
        restart_supported: bool,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            self._game = game
            self._model_dir = model_dir.resolve()
            self._restart_supported = restart_supported
            self._on_reset = on_reset
            self._pending = None

    def request_reset(self, game: str) -> ModelResetRequest:
        with self._lock:
            if not self._restart_supported or self._game is None or self._model_dir is None:
                raise RuntimeError("The active model does not support an in-process restart.")
            if game != self._game:
                raise ValueError("The requested game does not match the active training run.")
            if self._pending is None:
                self._pending = ModelResetRequest(
                    game=self._game,
                    model_dir=self._model_dir,
                    on_reset=self._on_reset,
                )
            return self._pending

    def reset_requested(self) -> bool:
        with self._lock:
            return self._pending is not None

    def consume_reset(self) -> ModelResetRequest | None:
        with self._lock:
            request = self._pending
            self._pending = None
            return request

    def metadata(self) -> dict[str, object]:
        with self._lock:
            return {
                "game": self._game,
                "restart_supported": self._restart_supported,
                "reset_pending": self._pending is not None,
            }


_control = TrainingControl()


def configure_training_control(
    *,
    game: str,
    model_dir: Path,
    restart_supported: bool,
    on_reset: Callable[[], None] | None = None,
) -> None:
    _control.configure(
        game=game,
        model_dir=model_dir,
        restart_supported=restart_supported,
        on_reset=on_reset,
    )


def request_model_reset(game: str) -> ModelResetRequest:
    return _control.request_reset(game)


def model_reset_requested() -> bool:
    return _control.reset_requested()


def consume_model_reset() -> ModelResetRequest | None:
    return _control.consume_reset()


def control_metadata() -> dict[str, object]:
    return _control.metadata()


def perform_model_reset(request: ModelResetRequest) -> None:
    def delete_training_artifacts() -> None:
        if request.model_dir.is_dir():
            shutil.rmtree(request.model_dir)
        else:
            request.model_dir.unlink(missing_ok=True)
        if request.on_reset is not None:
            request.on_reset()

    get_store().reset_for_restart(delete_training_artifacts)
    logger.warning(f"Deleted model directory for {request.game}: {request.model_dir}")
