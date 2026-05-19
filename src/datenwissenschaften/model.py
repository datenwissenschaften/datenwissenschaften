from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from stable_baselines3 import PPO

from datenwissenschaften.console import ui_info, ui_warning
from datenwissenschaften.core.protocols import ModelBuilder as ModelFactory
from datenwissenschaften.core.protocols import TrainableModel

ModelLoader = Callable[..., TrainableModel]


def get_model_path(models_dir: str, game: str) -> str:
    path = os.path.join(models_dir, game, "model")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_metadata(model: Any) -> dict[str, Any]:
    return {
        "action_space": str(model.action_space),
        "batch_size": getattr(model, "batch_size", None),
        "clip_range": str(getattr(model, "clip_range", None)),
        "clip_range_vf": str(getattr(model, "clip_range_vf", None)),
        "device": str(getattr(model, "device", None)),
        "ent_coef": getattr(model, "ent_coef", None),
        "gae_lambda": getattr(model, "gae_lambda", None),
        "gamma": getattr(model, "gamma", None),
        "lr_schedule": str(getattr(model, "lr_schedule", None)),
        "max_grad_norm": getattr(model, "max_grad_norm", None),
        "n_envs": getattr(model, "n_envs", None),
        "n_epochs": getattr(model, "n_epochs", None),
        "n_steps": getattr(model, "n_steps", None),
        "normalize_advantage": getattr(model, "normalize_advantage", None),
        "num_timesteps": getattr(model, "num_timesteps", None),
        "observation_space": str(model.observation_space),
        "policy_class": str(getattr(model, "policy_class", None)),
        "policy_kwargs": getattr(model, "policy_kwargs", None),
        "sde_sample_freq": getattr(model, "sde_sample_freq", None),
        "seed": getattr(model, "seed", None),
        "target_kl": getattr(model, "target_kl", None),
        "use_sde": getattr(model, "use_sde", None),
        "verbose": getattr(model, "verbose", None),
        "vf_coef": getattr(model, "vf_coef", None),
        "_total_timesteps": getattr(model, "_total_timesteps", None),
    }


def load_or_create_model(
    venv: Any,
    *,
    build_model: ModelFactory,
    load_model: ModelLoader = PPO.load,
) -> TrainableModel:
    game = _required_env("RETRO_SPEEDLAB_GAME_ID")
    models_dir = _required_env("RETRO_SPEEDLAB_MODEL_DIR")
    model_path = get_model_path(models_dir, game)
    model_zip_path = f"{model_path}.zip"

    if not os.path.exists(model_zip_path):
        ui_info(f"No existing model found at {model_zip_path}. Starting fresh training session.")
        return build_model(venv)

    try:
        ui_info(f"Loading existing model: {model_path}.zip")
        return load_model(model_path, env=venv, verbose=0)
    except Exception as error:
        ui_warning(f"Failed to load model: {error}")
        ui_info("Starting fresh training session.")
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
