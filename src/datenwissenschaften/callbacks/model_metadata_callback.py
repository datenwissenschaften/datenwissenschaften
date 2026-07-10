from __future__ import annotations

import time

from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.ui.telemetry import publish_metadata


class ModelMetadataCallback(BaseCallback):
    def __init__(self, interval_seconds: float = 1.0) -> None:
        super().__init__()
        self.interval_seconds = max(0.1, float(interval_seconds))
        self._next_publish_at = 0.0

    def _on_training_start(self) -> None:
        self._publish()

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now >= self._next_publish_at:
            self._publish(now)
        return True

    def _on_rollout_end(self) -> None:
        self._publish()

    def _publish(self, now: float | None = None) -> None:
        runtime = get_runtime()
        publish_metadata("model", runtime.get_model_metadata(self.model), replace=True)
        self._next_publish_at = (time.monotonic() if now is None else now) + self.interval_seconds
