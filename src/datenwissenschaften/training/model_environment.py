from collections.abc import Mapping
from functools import partial
from typing import Any

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv


class ModelEnvironment(gym.Env):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        super().reset(**kwargs)
        return self.observation_space.sample(), {}

    def step(self, action: Any) -> Any:
        raise RuntimeError("ModelEnvironment cannot be stepped")


def build_model_environments(
    observation_space: gym.Space,
    state_actions: Mapping[str, tuple[int, ...]],
) -> dict[str, DummyVecEnv]:
    if not state_actions:
        raise ValueError("Training requires at least one state action space")
    empty_states = [state for state, actions in state_actions.items() if not actions]
    if empty_states:
        names = ", ".join(sorted(empty_states))
        raise ValueError(f"State action spaces must not be empty: {names}")
    return {
        state: DummyVecEnv(
            [
                partial(
                    ModelEnvironment,
                    observation_space,
                    gym.spaces.Discrete(len(actions)),
                )
            ]
        )
        for state, actions in state_actions.items()
    }
