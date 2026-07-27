from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import stable_retro
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.retro.rom_importer import import_roms


def build_environment(wrapper: Callable[[Any], Any], config_path: str | Path) -> VecFrameStack:
    config = load_config(config_path)
    logger.info(
        "Building {} environment(s) for {} / {}",
        config.training.num_envs,
        config.training.game,
        config.training.savestate,
    )
    import_roms(config.paths.roms)
    factories = [
        partial(
            _create_environment,
            wrapper,
            config.training.game,
            config.training.savestate,
            config.paths.models / config.training.game / config.training.savestate,
            index,
        )
        for index in range(config.training.num_envs)
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
