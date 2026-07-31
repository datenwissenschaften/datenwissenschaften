import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from box import Box
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.models.path import model_directory

PUBLISH_INTERVAL_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 15.0
RETRO_FRAMES_PER_SECOND = 60.0


class RunnerStatsPublisher(BaseCallback):
    def __init__(self, config: Box) -> None:
        super().__init__()
        self.config: Box = config
        self.path: Path = model_directory(config) / "runner-stats.json"
        self.stats: dict[str, int | float] = _load_stats(self.path)
        self.next_publish_at: float = 0.0

    def _on_step(self) -> bool:
        completed = False
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                _record_episode(self.stats, info)
                completed = True
        if completed:
            _save_stats(self.path, self.stats)
        if completed and time.monotonic() >= self.next_publish_at:
            self._publish()
        return True

    def _on_training_end(self) -> None:
        if int(self.stats["timed_episodes"]) > 0:
            self._publish()

    def _publish(self) -> None:
        try:
            response = httpx.post(
                f"{self.config.upload.url}/runner/stats",
                headers={"X-API-Key": self.config.upload.api_key},
                json=_payload(self.config, self.stats),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            self.next_publish_at = time.monotonic() + PUBLISH_INTERVAL_SECONDS
        except httpx.HTTPError as error:
            logger.warning("Runner statistics upload failed: {}", error)


def _empty_stats() -> dict[str, int | float]:
    return {
        "wins": 0,
        "episodes": 0,
        "timed_episodes": 0,
        "training_seconds": 0.0,
        "latest_training_seconds": 0.0,
        "best_fitness": 0.0,
    }


def _load_stats(path: Path) -> dict[str, int | float]:
    if not path.is_file():
        return _empty_stats()
    stats = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stats, dict) or set(stats) != set(_empty_stats()):
        raise RuntimeError(f"Invalid runner statistics file: {path}")
    return stats


def _save_stats(path: Path, stats: dict[str, int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stats, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _record_episode(stats: dict[str, int | float], info: dict[str, Any]) -> None:
    episode = info["episode"]
    duration = float(episode["l"]) * int(info["action_repeat"]) / RETRO_FRAMES_PER_SECOND
    score = float(episode["r"])
    first_episode = int(stats["timed_episodes"]) == 0
    stats["wins"] = int(stats["wins"]) + int(bool(info["won"]))
    stats["episodes"] = max(int(stats["episodes"]), int(info["episode_number"]))
    stats["timed_episodes"] = int(stats["timed_episodes"]) + 1
    stats["training_seconds"] = float(stats["training_seconds"]) + duration
    stats["best_fitness"] = score if first_episode else max(float(stats["best_fitness"]), score)
    stats["latest_training_seconds"] = duration


def _payload(config: Box, stats: dict[str, int | float]) -> dict[str, str | int | float]:
    timed_episodes = int(stats["timed_episodes"])
    return {
        "runner_id": config.training.runner_id,
        "current_runner": config.training.runner_name,
        "current_game": config.training.game,
        "current_savestate": config.training.savestate,
        "best_fitness": float(stats["best_fitness"]),
        "wins": int(stats["wins"]),
        "episodes": int(stats["episodes"]),
        "average_training_seconds": float(stats["training_seconds"]) / timed_episodes,
        "latest_training_seconds": float(stats["latest_training_seconds"]),
        "source_updated_at": datetime.now(UTC).isoformat(),
    }
