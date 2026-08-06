import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback

from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import AdaptiveRecurrentPPO, load_agent
from datenwissenschaften.models.path import model_directory
from datenwissenschaften.rewards.normalizer import save_reward_normalizer
from datenwissenschaften.training.runner_stats import RunnerStatsPublisher
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader

CHECKPOINT_INTERVAL = 10_000


def train(environment: Any, config_path: str | Path) -> None:
    config = load_config(config_path)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    checkpoint = model_directory(config) / "model"
    model = load_agent(environment, checkpoint)
    uploader = WinningEpisodeUploader(config)
    stats_publisher = RunnerStatsPublisher(config)
    checkpoint_callback = ConvertCallback(
        _checkpoint_callback(model, checkpoint),
    )
    adaptation_callback = EpisodeAdaptationCallback()
    logger.info(
        "Training one shared agent for {} / {} with {} environment(s)",
        config.training.game,
        config.training.savestate,
        model.n_envs,
    )

    while True:
        model.learn(
            total_timesteps=CHECKPOINT_INTERVAL,
            reset_num_timesteps=False,
            callback=[uploader, stats_publisher, checkpoint_callback, adaptation_callback],
        )
        if uploader.completed:
            uploader.remove_model()
            logger.success("Removed completed agent")
            return


def _checkpoint_callback(model: Any, checkpoint: Path) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    next_checkpoint = [
        (model.num_timesteps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL,
    ]
    return partial(
        _save_checkpoint,
        checkpoint=checkpoint,
        next_checkpoint=next_checkpoint,
    )


def _save_checkpoint(
    locals_: dict[str, Any],
    globals_: dict[str, Any],
    *,
    checkpoint: Path,
    next_checkpoint: list[int],
) -> bool:
    del globals_
    model = locals_["self"]
    if model.num_timesteps < next_checkpoint[0]:
        return True

    atomic_save(model, checkpoint)
    save_reward_normalizer(model.get_env(), checkpoint.parent)
    logger.debug("Saved shared agent after {:,} environment steps", model.num_timesteps)
    next_checkpoint[0] = (model.num_timesteps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
    return True


class EpisodeAdaptationCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._returns: list[float] = []

    def _on_training_start(self) -> None:
        self._returns = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for index, (reward, done, info) in enumerate(zip(rewards, dones, infos, strict=True)):
            self._returns[index] += float(info.get("extrinsic_reward", reward))
            if not done:
                continue
            if isinstance(self.model, AdaptiveRecurrentPPO):
                self.model.record_episode_outcome(self._returns[index], bool(info.get("won", False)))
            self._returns[index] = 0.0
        return True
