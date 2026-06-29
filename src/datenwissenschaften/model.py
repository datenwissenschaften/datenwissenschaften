from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3 import PPO

from datenwissenschaften.core.protocols import ModelBuilder as ModelFactory
from datenwissenschaften.core.protocols import TrainableModel
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config

ModelLoader = Callable[..., TrainableModel]


def get_model_path(models_dir: str, game: str) -> str:
    path = os.path.join(models_dir, game, "model")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_metadata(model: Any) -> dict[str, Any]:
    return {
        "action_space": str(model.action_space),
        "class": _class_path(model),
        "device": str(getattr(model, "device", None)),
        "n_envs": getattr(model, "n_envs", None),
        "num_timesteps": getattr(model, "num_timesteps", None),
        "observation_space": str(model.observation_space),
        "total_timesteps": getattr(model, "_total_timesteps", None),
    }


def _class_path(obj: Any) -> str:
    if isinstance(obj, type):
        cls = obj
    elif callable(obj):
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if module and qualname:
            return f"{module}.{qualname}"

        cls = obj.__class__
    else:
        cls = obj.__class__

    return f"{cls.__module__}.{cls.__qualname__}"


def load_or_create_model(
    venv: Any,
    *,
    build_model: ModelFactory,
    load_model: ModelLoader = PPO.load,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> TrainableModel:
    config = load_config(config_path)
    model_path = get_model_path(str(config.paths.models_dir), config.training.game)
    model_zip_path = f"{model_path}.zip"

    if not os.path.exists(model_zip_path):
        logger.info(f"No existing model found at {model_zip_path}. Starting fresh training session.")
        return build_model(venv)

    try:
        logger.info(f"Loading existing model: {model_path}.zip")
        return load_model(model_path, env=venv, verbose=0)
    except Exception as error:
        logger.warning(f"Failed to load model: {error}")
        logger.info("Starting fresh training session.")
        return build_model(venv)


class ModelBuilder:
    def __init__(self, build_model: ModelFactory | type, *, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.build_model = build_model
        self.config_path = config_path

    def build(self, venv: Any) -> TrainableModel:
        build_model = self.build_model() if isinstance(self.build_model, type) else self.build_model
        load_model = getattr(build_model, "load", PPO.load)
        return load_or_create_model(
            venv,
            build_model=build_model,
            load_model=load_model,
            config_path=self.config_path,
        )
