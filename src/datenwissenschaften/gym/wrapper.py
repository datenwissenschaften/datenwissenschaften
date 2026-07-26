from pathlib import Path
from typing import Any, Generic, TypeVar

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

    def __init__(
        self,
        env: gym.Env,
        *,
        action_table: np.ndarray | None,
        model_dir: Path,
    ) -> None:
        super().__init__(env)
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.machine = StateMachine(self.start_state_cls(model_dir))
        self.state_types: tuple[type[State[T]], ...] = _state_types(self.start_state_cls, self.training_state_classes)
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(action_table)) if action_table is not None else env.action_space
        ram_size = sum(length for _, length in self.ram_info_cls.ram_map().values())
        if ram_size < 1:
            raise ValueError("Training requires at least one declared RAM field")
        self.observation_space = gym.spaces.Dict(
            {
                "image": env.observation_space,
                "features": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(ram_size + len(self.state_types) + 3,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        frame, info = self.env.reset(**kwargs)
        ram = _ram(self.ram_info_cls, self.env.unwrapped)
        self.machine.reset(ram, frame, self.start_state_cls)
        return _observation(frame, ram, self.state_types, self.machine.current), info

    def step(self, action: WrapperActType) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        reward = 0.0
        for _ in range(self.action_repeat):
            frame, _, terminated, truncated, info = self.env.step(_action(action, self.action_table, self.action_space))
            ram = _ram(self.ram_info_cls, self.env.unwrapped)
            state_reward, state_terminated, state_truncated = self.machine.step(ram, frame)
            reward += state_reward
            terminated = terminated or state_terminated
            truncated = truncated or state_truncated
            if terminated or truncated:
                break
        won = self.machine.current._won()
        terminated = terminated or won
        info.update(
            {
                "state": self.machine.name,
                "won": won,
                "ram": ram.to_dict(),
            }
        )
        if terminated or truncated:
            retro = self.env.unwrapped
            recording = Path(retro.movie_path) / (
                f"{retro.gamename}-{Path(retro.statename).stem}-{retro.movie_id - 1:06d}.bk2"
            )
            info["episode_bk2_path"] = str(recording)
        observation = _observation(frame, ram, self.state_types, self.machine.current)
        return observation, reward, terminated, truncated, info


def _ram(model: type[T], emulator: Any) -> T:
    return model.from_ram(emulator.get_ram())


def _state_types(start: type[State[T]], states: tuple[type[State[T]], ...]) -> tuple[type[State[T]], ...]:
    return tuple(dict.fromkeys((start, *states)))


def _observation(
    frame: np.ndarray,
    ram: T,
    states: tuple[type[State[T]], ...],
    current: State[T],
) -> dict[str, np.ndarray]:
    state = np.zeros(len(states), dtype=np.float32)
    state[states.index(type(current))] = 1.0
    template = np.zeros(3, dtype=np.float32)
    if current.target_detector is not None:
        template[0] = float(current.target_detector.seen)
        if current.target_detector.position is not None:
            height, width = current.frame.shape[:2]
            template[1] = current.target_detector.position[0] / width
            template[2] = current.target_detector.position[1] / height
    features = np.concatenate((np.asarray(ram.features(), dtype=np.float32), state, template))
    return {"image": frame, "features": features}


def _action(
    action: WrapperActType,
    action_table: np.ndarray | None,
    action_space: gym.Space,
) -> WrapperActType:
    if action_table is None:
        return action
    if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
        raise TypeError(f"Action must be an integer, got {type(action).__name__}")
    index = int(action)
    if not action_space.contains(index):
        raise ValueError(f"Action {index} is outside {action_space}")
    return action_table[index]
