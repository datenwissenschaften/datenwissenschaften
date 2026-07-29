import sys
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.utils import LinearSchedule

from datenwissenschaften.checkpoints.model import atomic_save, atomic_save_replay_buffer
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.models.path import model_directory
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader

TRAINING_CHUNK_STEPS = 10_000


def train(environment: Any, config_path: str | Path) -> None:
    config = load_config(config_path)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    checkpoint = model_directory(config) / "model"
    model = load_agent(environment, checkpoint)
    uploader = WinningEpisodeUploader(config)
    logger.info(
        "Training {} / {} with {} environment(s)",
        config.training.game,
        config.training.savestate,
        model.n_envs,
    )
    while True:
        logger.debug("Starting training chunk at {:,} environment steps", model.num_timesteps)
        _continue_exploration_schedule(model)
        model.learn(total_timesteps=TRAINING_CHUNK_STEPS, reset_num_timesteps=False, callback=uploader)
        atomic_save_replay_buffer(model, checkpoint.with_suffix(".replay.pkl"))
        atomic_save(model, checkpoint)
        logger.debug("Saved agent after {:,} environment steps", model.num_timesteps)


def _continue_exploration_schedule(model: Any) -> None:
    if model.num_timesteps == 0:
        return
    model.exploration_schedule = LinearSchedule(
        model.exploration_rate,
        model.exploration_final_eps,
        1.0,
    )
