from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3 import PPO

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.core.protocols import ModelBuilder as ModelFactory
from datenwissenschaften.core.protocols import TrainableModel
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config

ModelLoader = Callable[..., TrainableModel]


def get_model_path(models_dir: str, game: str) -> str:
    path = os.path.join(models_dir, game, "model")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_metadata(model: Any) -> dict[str, Any]:
    metadata = {
        "action_space": str(getattr(model, "action_space", None)),
        "class": f"{model.__class__.__module__.partition('.')[0]}.{model.__class__.__name__}",
        "class_path": _class_path(model),
        "device": str(getattr(model, "device", None)),
        "n_envs": getattr(model, "n_envs", None),
        "num_timesteps": getattr(model, "num_timesteps", None),
        "observation_space": str(getattr(model, "observation_space", None)),
        "total_timesteps": getattr(model, "_total_timesteps", None),
    }
    ppo_fields = (
        "batch_size",
        "n_steps",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "use_sde",
        "sde_sample_freq",
        "learning_rate",
        "policy_kwargs",
    )
    ppo = {field: getattr(model, field) for field in ppo_fields if hasattr(model, field)}
    if ppo:
        ppo["policy"] = _class_path(model.policy) if getattr(model, "policy", None) is not None else None
        metadata["ppo"] = ppo
    rnd_fields = (
        "rnd_output_size",
        "rnd_learning_rate",
        "rnd_update_proportion",
        "rnd_gamma",
        "rnd_intrinsic_coefficient",
        "rnd_final_intrinsic_coefficient",
        "rnd_anneal_steps",
        "rnd_reward_clip",
    )
    rnd = {field: getattr(model, field) for field in rnd_fields if hasattr(model, field)}
    if rnd:
        active_rnd = getattr(model, "rnd", None)
        rnd["observations_seen"] = int(active_rnd.observations_seen.item()) if active_rnd is not None else 0
        rnd["current_intrinsic_coefficient"] = active_rnd.coefficient if active_rnd is not None else None
        metadata["rnd"] = rnd
    return metadata


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
    configure_accelerator()
    config = load_config(config_path)
    model_path = get_model_path(str(config.paths.models_dir), config.training.game)
    model_zip_path = f"{model_path}.zip"
    cleanup = getattr(build_model, "cleanup_incompatible_artifacts", None)
    if callable(cleanup):
        cleanup(config)

    if not os.path.exists(model_zip_path):
        logger.info(f"No existing model found at {model_zip_path}. Starting fresh training session.")
        return build_model(venv)

    try:
        logger.info(f"Loading existing model: {model_path}.zip")
        return load_model(model_path, env=venv, verbose=0)
    except Exception as error:
        logger.warning(f"Failed to load model: {error}")
        logger.info("Starting fresh training session.")
        Path(model_zip_path).unlink(missing_ok=True)
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
