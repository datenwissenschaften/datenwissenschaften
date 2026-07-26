from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    training_state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_repeat: int
    grayscale: bool

    def __init__(
        self,
        env: gym.Env,
        *,
        obs_size: tuple[int, int],
        action_table: np.ndarray | None,
        model_dir: Path,
    ) -> None:
        super().__init__(env)
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.machine = StateMachine(self.start_state_cls(model_dir))
        self.obs_size = obs_size
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(action_table)) if action_table is not None else env.action_space
        channels = 1 if self.grayscale else 3
        state_count = len(self._state_types())
        ram_size = sum(length for _, length in self.ram_info_cls.ram_map().values())
        self.observation_space = gym.spaces.Dict(
            {
                "visual": gym.spaces.Box(0, 255, shape=(channels, *obs_size), dtype=np.uint8),
                "ram": gym.spaces.Box(0.0, 1.0, shape=(ram_size,), dtype=np.float32),
                "state": gym.spaces.Box(0.0, 1.0, shape=(state_count,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        frame, info = self.env.reset(**kwargs)
        ram = self._ram()
        observation = self._image(frame)
        self.machine.reset(ram, frame, observation)
        return self._observation(observation, ram), info

    def step(self, action: WrapperActType) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        reward = 0.0
        for _ in range(self.action_repeat):
            frame, _, terminated, truncated, info = self.env.step(self._action(action))
            ram = self._ram()
            observation = self._image(frame)
            state_reward, state_terminated, state_truncated = self.machine.step(ram, frame, observation)
            reward += state_reward
            terminated = terminated or state_terminated
            truncated = truncated or state_truncated
            if terminated or truncated:
                break
        won = self.machine.current._won()
        terminated = terminated or won
        info.update({"state": self.machine.name, "won": won, "ram": ram.to_dict()})
        if terminated or truncated:
            retro = self.env.unwrapped
            recording = Path(retro.movie_path) / (
                f"{retro.gamename}-{Path(retro.statename).stem}-{retro.movie_id - 1:06d}.bk2"
            )
            info["episode_bk2_path"] = str(recording)
        return self._observation(observation, ram), reward, terminated, truncated, info

    def _observation(self, image: np.ndarray, ram: T) -> dict[str, np.ndarray]:
        state = np.zeros(len(self._state_types()), dtype=np.float32)
        state[self._state_types().index(type(self.machine.current))] = 1.0
        return {"visual": image, "ram": np.asarray(ram.features(), dtype=np.float32), "state": state}

    def _action(self, action: WrapperActType) -> WrapperActType:
        if self.action_table is None:
            return action
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise TypeError(f"Action must be an integer, got {type(action).__name__}")
        index = int(action)
        if not self.action_space.contains(index):
            raise ValueError(f"Action {index} is outside {self.action_space}")
        return self.action_table[index]

    def _ram(self) -> T:
        return self.ram_info_cls.from_ram(self.env.unwrapped.get_ram())

    def _image(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(frame, self.obs_size[::-1], interpolation=cv2.INTER_AREA)
        return resized[None, ...] if self.grayscale else np.transpose(resized, (2, 0, 1))

    def _state_types(self) -> tuple[type[State[T]], ...]:
        return tuple(dict.fromkeys((self.start_state_cls, *self.training_state_classes)))
