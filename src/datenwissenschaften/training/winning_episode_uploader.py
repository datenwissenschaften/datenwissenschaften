from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from box import Box
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.models.path import model_directory
from datenwissenschaften.rewards.normalizer import remove_reward_normalizer


class WinningEpisodeUploader(BaseCallback):
    def __init__(self, config: Box) -> None:
        super().__init__()
        self.config: Box = config
        self.completed: bool = False
        self.upload_failed: bool = False

    def _on_step(self) -> bool:
        self.upload_failed = False
        self.completed = self.process(self.locals["dones"], self.locals["infos"])
        if self.completed:
            self.remove_model()
        return not self.completed and not self.upload_failed

    def process(self, dones: np.ndarray, infos: list[dict[str, Any]]) -> bool:
        completed = False
        for done, info in zip(dones, infos, strict=True):
            if done:
                episode_completed, upload_failed = _process_episode(self.config, info)
                completed = episode_completed or completed
                self.upload_failed = upload_failed or self.upload_failed
        return completed

    def remove_model(self) -> None:
        root = model_directory(self.config)
        checkpoints = (*root.glob("*/model.zip"), root / "model.zip")
        for checkpoint in checkpoints:
            checkpoint.unlink(missing_ok=True)
        remove_reward_normalizer(root)


def _process_episode(config: Box, info: dict[str, Any]) -> tuple[bool, bool]:
    recording = Path(info["episode_bk2_path"])
    root = model_directory(config)
    reward_path = root / "best.score"
    episode = info["episode"]
    score = float(episode["r"])
    new_best = _record_score(reward_path, score)
    best_score = float(reward_path.read_text(encoding="utf-8"))
    logger.debug(
        "Episode finished: reward={:.3f}, best={:.3f}, steps={}, end={}, won={}",
        episode["r"],
        best_score,
        episode["l"],
        info["state"],
        info["won"],
    )
    if new_best and not info["won"]:
        logger.debug("New best training score: {:.3f}", score)
        if not _upload_episode(_upload_training, config, info, recording):
            return False, True
    if not info["won"]:
        recording.unlink(missing_ok=True)
        return False, False
    if not _upload_episode(_upload_win, config, info, recording):
        return False, True
    return _curriculum_complete(info), False


def _curriculum_complete(info: dict[str, Any]) -> bool:
    if "full_run" in info:
        return bool(info["full_run"] and info["won"])
    if "curriculum_complete" in info:
        return False
    return bool(info["won"])


def _upload_episode(
    upload: Callable[[Box, dict[str, Any], Path], None],
    config: Box,
    info: dict[str, Any],
    recording: Path,
) -> bool:
    try:
        upload(config, info, recording)
    except httpx.HTTPError as error:
        logger.warning("Upload failed for {}: {}", recording.name, error)
        return False
    return True


def _record_score(path: Path, score: float) -> bool:
    if path.is_file():
        best = float(path.read_text(encoding="utf-8"))
        if score <= best:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(repr(score), encoding="utf-8")
    return True


def _upload_training(
    config: Box,
    info: dict[str, Any],
    recording: Path,
) -> None:
    with recording.open("rb") as stream:
        _post(
            config,
            info,
            "TRAINING",
            {"bk2_file": (recording.name, stream, "application/zip")},
        )


def _upload_win(
    config: Box,
    info: dict[str, Any],
    recording: Path,
) -> None:
    logger.info("Uploading winning episode {}", recording.name)
    with recording.open("rb") as stream:
        _post(
            config,
            info,
            "WON",
            {"bk2_file": (recording.name, stream, "application/zip")},
        )
    recording.unlink(missing_ok=True)
    logger.success("Uploaded winning episode {}", recording.name)


def _post(
    config: Box,
    info: dict[str, Any],
    run_type: str,
    files: dict[str, Any],
) -> None:
    data: dict[str, Any] = {
        "game": config.training.game,
        "category": config.training.savestate,
        "action_repeat": info["action_repeat"],
        "episode_number": info["episode_number"],
        "type": run_type,
    }
    if "curriculum_state" in info:
        curriculum = info["curriculum_state"]
        data["curriculum"] = curriculum
        data["progress"] = _curriculum_progress(info, curriculum)
    response = httpx.post(
        f"{config.upload.url}/runs",
        headers={"X-API-Key": config.upload.api_key},
        data=data,
        files=files,
    )
    response.raise_for_status()


def _curriculum_progress(info: dict[str, Any], curriculum: str) -> float:
    if "curriculum_progress" not in info:
        return 0.0
    progress = info["curriculum_progress"]
    if not isinstance(progress, dict) or curriculum not in progress:
        return 0.0
    state_progress = progress[curriculum]
    if not isinstance(state_progress, dict):
        return 0.0
    wins = state_progress["wins"]
    target = state_progress["win_target"]
    return float(wins) / float(target)
