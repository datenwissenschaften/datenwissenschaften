import sys
from pathlib import Path
from typing import Any

from loguru import logger

from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader


def train(environment: Any, config_path: str | Path) -> None:
    config = load_config(config_path)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    checkpoint = config.paths.models / config.training.game / config.training.savestate / "ppo-v4"
    model = load_agent(environment, checkpoint)
    uploader = WinningEpisodeUploader(config)
    logger.info(
        "Training {} / {} with {} environment(s)",
        config.training.game,
        config.training.savestate,
        model.n_envs,
    )
    while True:
        logger.debug("Starting rollout at {:,} environment steps", model.num_timesteps)
        model.learn(total_timesteps=model.n_steps * model.n_envs, reset_num_timesteps=False, callback=uploader)
        atomic_save(model, checkpoint)
        logger.debug("Saved agent after {:,} environment steps", model.num_timesteps)
