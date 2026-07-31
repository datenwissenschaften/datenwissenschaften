import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import ConvertCallback

from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.models.path import model_directory
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader

CHECKPOINT_INTERVAL = 10_000


def train(environment: Any, config_path: str | Path) -> None:
    config = load_config(config_path)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    checkpoint = model_directory(config) / "model"
    model = load_agent(environment, checkpoint)
    uploader = WinningEpisodeUploader(config)
    checkpoint_callback = ConvertCallback(
        _checkpoint_callback(model, checkpoint),
    )
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
            callback=[uploader, checkpoint_callback],
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
    logger.debug("Saved shared agent after {:,} environment steps", model.num_timesteps)
    next_checkpoint[0] = (model.num_timesteps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
    return True
