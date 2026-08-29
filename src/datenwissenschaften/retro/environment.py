from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import stable_retro
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.path import model_directory
from datenwissenschaften.retro.rom_importer import import_roms
from datenwissenschaften.rewards.normalizer import normalize_rewards


def build_environment(wrapper: Callable[[Any], Any], config_path: str | Path) -> VecNormalize:
    config = load_config(config_path)
    logger.info(
        "Building one CPU environment for {} / {}",
        config.training.game,
        config.training.savestate,
    )
    import_roms(config.paths.roms)
    models_path = model_directory(config)
    factory = partial(
        _create_environment,
        wrapper,
        config.training.game,
        config.training.savestate,
        models_path,
        0,
    )
    environments = DummyVecEnv([factory])
    logger.success("Environments ready")
    return normalize_rewards(VecMonitor(environments), models_path)


def _create_environment(wrapper: Callable[[Any], Any], game: str, savestate: str, model_dir: Path, index: int) -> Any:
    recordings = model_dir / "episodes" / str(index)
    recordings.mkdir(parents=True, exist_ok=True)
    return wrapper(
        stable_retro.make(game, savestate, render_mode="rgb_array", record=recordings),
        model_dir=model_dir,
    )
