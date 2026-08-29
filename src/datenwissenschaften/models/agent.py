from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file

from datenwissenschaften.rewards.normalizer import REWARD_DISCOUNT_FACTOR

CHECKPOINT_VERSION = 7
CPU_THREADS = 1
CPU_INTEROP_THREADS = 1
ROLLOUT_STEPS = 128
HIDDEN_SIZE = 32


def load_agent(environment: Any, path: Path) -> A2C:
    _configure_cpu()
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        _validate_checkpoint(checkpoint)
        logger.info("Loading state agent from {}", checkpoint)
        model = A2C.load(checkpoint, env=environment, device="cpu")
        _normalize_binary_action_storage(model)
        model.set_logger(configure(folder=None, format_strings=[]))
        return model
    logger.info("Creating compact A2C state agent")
    model = A2C(
        "MlpPolicy",
        environment,
        device="cpu",
        learning_rate=0.0007,
        n_steps=ROLLOUT_STEPS,
        gamma=REWARD_DISCOUNT_FACTOR,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs={"net_arch": {"pi": [HIDDEN_SIZE], "vf": [HIDDEN_SIZE]}},
    )
    model.checkpoint_version = CHECKPOINT_VERSION
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def _configure_cpu() -> None:
    torch.set_num_threads(CPU_THREADS)
    if torch.get_num_interop_threads() != CPU_INTEROP_THREADS:
        torch.set_num_interop_threads(CPU_INTEROP_THREADS)


def _normalize_binary_action_storage(model: A2C) -> None:
    model.action_space.dtype = np.dtype(np.float32)
    model.rollout_buffer.action_space = model.action_space
    model.rollout_buffer.actions = model.rollout_buffer.actions.astype(np.float32)


def _validate_checkpoint(checkpoint: Path) -> None:
    data, _, _ = load_from_zip_file(checkpoint, device="cpu")
    if "checkpoint_version" not in data or data["checkpoint_version"] != CHECKPOINT_VERSION:
        raise RuntimeError(f"Unsupported checkpoint version: {checkpoint}")
    if "policy_class" not in data or data["policy_class"] is not ActorCriticPolicy:
        raise RuntimeError(f"Unsupported checkpoint algorithm: {checkpoint}")
