import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update

from datenwissenschaften.checkpoints.model import atomic_save, atomic_save_replay_buffer
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.models.path import model_directory
from datenwissenschaften.training.model_environment import build_model_environments
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader

CHECKPOINT_INTERVAL = 10_000
EXPLORATION_STEPS = 100_000


def train(environment: Any, config_path: str | Path) -> None:
    config = load_config(config_path)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    state_names = _state_names(environment)
    state_actions = _state_actions(environment, state_names)
    model_environments = build_model_environments(
        environment.observation_space,
        state_actions,
    )
    root = model_directory(config)
    models = {state: load_agent(model_environments[state], root / state / "model") for state in state_names}
    uploader = WinningEpisodeUploader(config)
    observations = environment.reset()
    states = _current_states(environment, state_names)
    environment_steps = sum(model.num_timesteps for model in models.values())
    next_checkpoint = (environment_steps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
    logger.info("Training {} state model(s) with {} environment(s)", len(models), environment.num_envs)

    while True:
        actions, model_actions = _actions(models, states, observations, state_actions)
        next_observations, rewards, dones, infos = environment.step(actions)
        next_states = _current_states(environment, state_names)
        _learn(models, states, observations, model_actions, next_observations, rewards, dones, infos)
        uploader.process(dones, infos)
        environment_steps += environment.num_envs
        observations = next_observations
        states = next_states

        if environment_steps >= next_checkpoint:
            _save(models, root)
            logger.debug("Saved state agents after {:,} environment steps", environment_steps)
            next_checkpoint = (environment_steps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL


def _state_names(environment: Any) -> tuple[str, ...]:
    names = tuple(environment.get_attr("state_names")[0])
    if not names:
        raise RuntimeError("Training requires at least one state")
    if len(names) != len(set(names)):
        raise RuntimeError("Training state names must be unique")
    return names


def _current_states(
    environment: Any,
    state_names: tuple[str, ...],
) -> tuple[str, ...]:
    states = tuple(environment.get_attr("state_name"))
    unknown_states = set(states) - set(state_names)
    if unknown_states:
        names = ", ".join(sorted(unknown_states))
        raise RuntimeError(f"Environment returned unknown training states: {names}")
    return states


def _state_actions(
    environment: Any,
    states: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    action_definitions = tuple(environment.get_attr("state_actions"))
    if not action_definitions:
        raise RuntimeError("Training requires at least one environment")
    actions = action_definitions[0]
    if any(definition != actions for definition in action_definitions[1:]):
        raise RuntimeError("State actions must match across environments")
    if set(actions) != set(states):
        raise RuntimeError("State actions must match training states")
    return {state: tuple(actions[state]) for state in states}


def _actions(
    models: Mapping[str, DQN],
    states: tuple[str, ...],
    observations: np.ndarray,
    state_actions: Mapping[str, tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    if set(models) != set(state_actions):
        raise RuntimeError("Models must match state actions")
    unknown_states = set(states) - set(models)
    if unknown_states:
        names = ", ".join(sorted(unknown_states))
        raise RuntimeError(f"No model exists for training states: {names}")
    actions = np.empty(len(states), dtype=np.int64)
    model_actions = np.empty(len(states), dtype=np.int64)
    for state in dict.fromkeys(states):
        indices = np.flatnonzero(np.asarray(states) == state)
        model = models[state]
        predicted, _ = model.predict(observations[indices], deterministic=True)
        predicted_actions = np.asarray(predicted)
        _validate_model_actions(state, predicted_actions, len(indices), len(state_actions[state]))
        model_actions[indices] = predicted_actions
        exploration_rate = _exploration_rate(model)
        random_indices = indices[np.random.random(len(indices)) < exploration_rate]
        for index in random_indices:
            model_actions[index] = np.random.randint(len(state_actions[state]))
        actions[indices] = np.asarray(state_actions[state])[model_actions[indices]]
    return actions, model_actions


def _validate_model_actions(
    state: str,
    actions: np.ndarray,
    environment_count: int,
    action_count: int,
) -> None:
    if actions.shape != (environment_count,):
        raise RuntimeError(f"{state} model returned actions with shape {actions.shape}")
    if not np.issubdtype(actions.dtype, np.integer) or np.issubdtype(actions.dtype, np.bool_):
        raise RuntimeError(f"{state} model returned non-integer actions")
    if np.any(actions < 0) or np.any(actions >= action_count):
        raise RuntimeError(f"{state} model returned an action outside its state action space")


def _exploration_rate(model: DQN) -> float:
    progress = min(model.num_timesteps / EXPLORATION_STEPS, 1.0)
    rate = model.exploration_initial_eps + progress * (model.exploration_final_eps - model.exploration_initial_eps)
    model.exploration_rate = rate
    return rate


def _learn(
    models: Mapping[str, DQN],
    states: tuple[str, ...],
    observations: np.ndarray,
    actions: np.ndarray,
    next_observations: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, Any]],
) -> None:
    trained: dict[str, int] = {}
    for index, state in enumerate(states):
        info = infos[index]
        transitioned = info["state"] != state
        terminal = bool(dones[index] or transitioned)
        next_observation = info["terminal_observation"] if dones[index] else next_observations[index]
        model = models[state]
        model.replay_buffer.add(
            observations[index][None],
            next_observation[None],
            np.asarray([actions[index]]),
            np.asarray([rewards[index]]),
            np.asarray([terminal]),
            [info],
        )
        model.num_timesteps += 1
        trained[state] = trained.get(state, 0) + 1

    for state, transitions in trained.items():
        model = models[state]
        gradient_steps = min(transitions, max(0, model.num_timesteps - model.learning_starts))
        if gradient_steps:
            model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)
        if (
            model.num_timesteps // model.target_update_interval
            != (model.num_timesteps - transitions) // model.target_update_interval
        ):
            polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)


def _save(models: Mapping[str, DQN], root: Path) -> None:
    for state, model in models.items():
        checkpoint = root / state / "model"
        atomic_save_replay_buffer(model, checkpoint.with_suffix(".replay.pkl"))
        atomic_save(model, checkpoint)
