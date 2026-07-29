import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import stable_retro
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.path import model_directory
from datenwissenschaften.retro.rom_importer import import_roms


def build_environment(wrapper: Callable[[Any], Any], config_path: str | Path) -> VecFrameStack:
    config = load_config(config_path)
    environment_count = len(os.sched_getaffinity(0))
    if environment_count < 1:
        raise RuntimeError("The process must have at least one available CPU")
    logger.info(
        "Building {} environment(s) for {} / {}",
        environment_count,
        config.training.game,
        config.training.savestate,
    )
    import_roms(config.paths.roms)
    models_path = model_directory(config)
    factories = [
        partial(
            _create_environment,
            wrapper,
            config.training.game,
            config.training.savestate,
            models_path,
            index,
        )
        for index in range(environment_count)
    ]
    environments = SubprocVecEnv(factories) if len(factories) > 1 else DummyVecEnv(factories)
    logger.success("Environments ready")
    return VecFrameStack(VecMonitor(environments), n_stack=4)


def _create_environment(wrapper: Callable[[Any], Any], game: str, savestate: str, model_dir: Path, index: int) -> Any:
    recordings = model_dir / "episodes" / str(index)
    recordings.mkdir(parents=True, exist_ok=True)
    return wrapper(
        stable_retro.make(game, savestate, render_mode="rgb_array", record=recordings),
        model_dir=model_dir,
    )
