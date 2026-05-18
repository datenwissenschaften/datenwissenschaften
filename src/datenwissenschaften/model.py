from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from loguru import logger
from stable_baselines3 import PPO

from datenwissenschaften.core.protocols import ModelBuilder as ModelFactory
from datenwissenschaften.core.protocols import TrainableModel

ModelLoader = Callable[..., TrainableModel]


def get_model_path(models_dir: str, game: str) -> str:
    path = os.path.join(models_dir, game, "model")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_or_create_model(
    venv: Any,
    *,
    build_model: ModelFactory,
    load_model: ModelLoader = PPO.load,
) -> TrainableModel:
    game = _required_env("RETRO_ARENA_GAME_ID")
    models_dir = _required_env("RETRO_ARENA_MODEL_DIR")
    model_path = get_model_path(models_dir, game)
    model_zip_path = f"{model_path}.zip"

    if not os.path.exists(model_zip_path):
        logger.info("No existing model found at {}. Starting fresh training session.", model_zip_path)
        return build_model(venv)

    try:
        logger.info("Loading existing model: {}.zip", model_path)
        return load_model(model_path, env=venv, verbose=0)
    except Exception as error:
        logger.warning("Failed to load model: {}", error)
        logger.info("Starting fresh training session.")
        return build_model(venv)


class ModelBuilder:
    def __init__(
        self,
        build_model: ModelFactory | type,
        *,
        load_model: ModelLoader = PPO.load,
    ) -> None:
        self.build_model = build_model
        self.load_model = load_model

    def build(self, venv: Any) -> TrainableModel:
        build_model = self.build_model() if isinstance(self.build_model, type) else self.build_model
        return load_or_create_model(
            venv,
            build_model=build_model,
            load_model=self.load_model,
        )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set.")
    return value
