from typing import Any

import gymnasium as gym


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
