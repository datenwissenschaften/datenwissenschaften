from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from stable_baselines3 import PPO

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.core.protocols import ModelBuilder as ModelFactory
from datenwissenschaften.core.protocols import TrainableModel
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config

ModelLoader = Callable[..., TrainableModel]


def model_parameters_are_finite(model: Any) -> bool:
    get_parameters = getattr(model, "get_parameters", None)
    if not callable(get_parameters):
        return True

    def tensors(value: Any):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from tensors(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from tensors(nested)

    return all(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()) for tensor in tensors(get_parameters())
    )


def get_model_path(models_dir: str, game: str, state_name: str) -> str:
    path = os.path.join(models_dir, game, state_name, "model")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_model_metadata(model: Any) -> dict[str, Any]:
    metadata = {
        "action_space": str(getattr(model, "action_space", None)),
        "class": f"{model.__class__.__module__.partition('.')[0]}.{model.__class__.__name__}",
        "class_path": _class_path(model),
        "display_name": getattr(model, "display_name", None),
        "description": getattr(model, "description", None),
        "device": str(getattr(model, "device", None)),
        "n_envs": getattr(model, "n_envs", None),
        "num_timesteps": getattr(model, "num_timesteps", None),
        "observation_space": str(getattr(model, "observation_space", None)),
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
        rnd["adaptation_multiplier"] = getattr(model, "adaptation_multiplier", None)
        rnd["adaptation_reason"] = getattr(model, "adaptation_reason", None)
        rnd["adaptive_autoconfigure"] = getattr(model, "adaptive_autoconfigure", None)
        rnd["adaptive_action_count"] = getattr(model, "adaptive_action_count", None)
        rnd["adaptive_observation_pixels"] = getattr(model, "adaptive_observation_pixels", None)
        rnd["adaptive_rollout_steps"] = getattr(model, "adaptive_rollout_steps", None)
        rnd["adaptive_score_delta"] = getattr(model, "adaptive_score_delta", None)
        rnd["adaptive_score_staleness_episodes"] = getattr(model, "adaptive_score_staleness_episodes", None)
        rnd["adaptive_no_win_staleness_episodes"] = getattr(model, "adaptive_no_win_staleness_episodes", None)
        rnd["adaptive_multiplier_min"] = getattr(model, "adaptive_multiplier_min", None)
        rnd["adaptive_multiplier_max"] = getattr(model, "adaptive_multiplier_max", None)
        rnd["adaptive_learning_rate_min"] = getattr(model, "adaptive_learning_rate_min", None)
        rnd["adaptive_learning_rate_max"] = getattr(model, "adaptive_learning_rate_max", None)
        rnd["adaptive_rnd_update_proportion"] = getattr(model, "adaptive_rnd_update_proportion", None)
        rnd["adaptive_rnd_update_max"] = getattr(model, "adaptive_rnd_update_max", None)
        rnd["episodes_since_score_improvement"] = getattr(model, "episodes_since_score_improvement", None)
        rnd["episodes_since_win"] = getattr(model, "episodes_since_win", None)
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
    state_name: str,
) -> TrainableModel:
    configure_accelerator()
    config = load_config(config_path)
    model_path = get_model_path(str(config.paths.models_dir), config.training.game_identity, state_name)
    model_zip_path = f"{model_path}.zip"
    cleanup = getattr(build_model, "cleanup_incompatible_artifacts", None)
    if callable(cleanup):
        cleanup(config)

    if not os.path.exists(model_zip_path):
        logger.info(f"No existing model found at {model_zip_path}. Starting fresh training session.")
        return build_model(venv)

    try:
        logger.info(f"Loading existing model: {model_path}.zip")
        model = load_model(model_path, env=venv, verbose=0)
        if not model_parameters_are_finite(model):
            raise ValueError("model checkpoint contains non-finite parameters")
        return model
    except Exception as error:
        logger.warning(f"Failed to load model: {error}")
        logger.info("Starting fresh training session.")
        Path(model_zip_path).unlink(missing_ok=True)
    return build_model(venv)


class ModelBuilder:
    def __init__(self, build_model: ModelFactory | type, *, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.build_model = build_model
        self.config_path = config_path

    def build(self, venv: Any, *, state_name: str) -> TrainableModel:
        build_model = self.build_model() if isinstance(self.build_model, type) else self.build_model
        load_model = getattr(build_model, "load", PPO.load)
        return load_or_create_model(
            venv,
            build_model=build_model,
            load_model=load_model,
            config_path=self.config_path,
            state_name=state_name,
        )
