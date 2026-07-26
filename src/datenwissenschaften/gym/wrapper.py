from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType
from loguru import logger

from datenwissenschaften.curriculum.progress import SavestateCurriculum
from datenwissenschaften.geometry.features import FEATURE_COUNT
from datenwissenschaften.geometry.memory import TileMemory
from datenwissenschaften.geometry.player import PlayerGeometry
from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    training_state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_repeat: int
    tile_size: int

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
        self.curriculum: SavestateCurriculum = SavestateCurriculum(
            model_dir / "curriculum-v5", tuple(state.__name__ for state in self.state_types)
        )
        self.tile_memory = TileMemory(model_dir / "geometry-v5.sqlite3", self.tile_size)
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(action_table)) if action_table is not None else env.action_space
        ram_size = sum(length for _, length in self.ram_info_cls.ram_map().values())
        if ram_size < 1:
            raise ValueError("Training requires at least one declared RAM field")
        feature_count = ram_size + len(self.state_types) + 3 + FEATURE_COUNT
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(feature_count,), dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        frame, info = self.env.reset(**kwargs)
        state_name, savestate = self.curriculum.start()
        if savestate is not None:
            frame = _restore(self.env.unwrapped, savestate)
        ram = _ram(self.ram_info_cls, self.env.unwrapped)
        player = ram.player_geometry()
        self.tile_memory.reset(frame, player)
        state_type = next(state for state in self.state_types if state.__name__ == state_name)
        self.machine.reset(ram, frame, state_type)
        return _observation(ram, player, self.state_types, self.machine.current, self.tile_memory), info

    def step(self, action: WrapperActType) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 0.0
        for _ in range(self.action_repeat):
            frame, _, terminated, truncated, info = self.env.step(_action(action, self.action_table, self.action_space))
            ram = _ram(self.ram_info_cls, self.env.unwrapped)
            previous_state = self.machine.name
            state_reward, state_terminated, state_truncated = self.machine.step(ram, frame)
            player = ram.player_geometry()
            dead = state_terminated and not self.machine.current._won()
            self.tile_memory.observe(frame, player, dead)
            if (
                self.machine.name != previous_state
                and not self.curriculum.recorded
                and previous_state == self.curriculum.episode_state
            ):
                if self.curriculum.transition(
                    previous_state, self.machine.name, bytes(self.env.unwrapped.em.get_state())
                ):
                    logger.success("Completed curriculum state {}", previous_state)
            reward += state_reward
            terminated = terminated or state_terminated
            truncated = truncated or state_truncated
            if terminated or truncated:
                break
        won = self.machine.current._won()
        if won and not self.curriculum.recorded and self.machine.name == self.curriculum.episode_state:
            if self.curriculum.victory(self.machine.name):
                logger.success("Completed curriculum state {}", self.machine.name)
        terminated = terminated or won
        info.update(
            {
                "state": self.machine.name,
                "won": won,
                "started_from_initial_state": self.curriculum.episode_state == self.state_types[0].__name__,
                "ram": ram.to_dict(),
            }
        )
        if terminated or truncated:
            retro = self.env.unwrapped
            recording = Path(retro.movie_path) / (
                f"{retro.gamename}-{Path(retro.statename).stem}-{retro.movie_id - 1:06d}.bk2"
            )
            info["episode_bk2_path"] = str(recording)
        observation = _observation(ram, player, self.state_types, self.machine.current, self.tile_memory)
        return observation, reward, terminated, truncated, info


def _restore(emulator: Any, savestate: bytes) -> np.ndarray:
    emulator.em.set_state(savestate)
    emulator.data.reset()
    emulator.data.update_ram()
    return emulator.get_screen(apply_rotation=True)


def _ram(model: type[T], emulator: Any) -> T:
    return model.from_ram(emulator.get_ram())


def _state_types(start: type[State[T]], states: tuple[type[State[T]], ...]) -> tuple[type[State[T]], ...]:
    return tuple(dict.fromkeys((start, *states)))


def _observation(
    ram: T,
    player: PlayerGeometry,
    states: tuple[type[State[T]], ...],
    current: State[T],
    tile_memory: TileMemory,
) -> np.ndarray:
    state = np.zeros(len(states), dtype=np.float32)
    state[states.index(type(current))] = 1.0
    template = np.zeros(3, dtype=np.float32)
    if current.target_detector is not None:
        template[0] = float(current.target_detector.seen)
        if current.target_detector.position is not None:
            height, width = current.frame.shape[:2]
            template[1] = current.target_detector.position[0] / width
            template[2] = current.target_detector.position[1] / height
    geometry = np.asarray(tile_memory.features(player), dtype=np.float32)
    return np.concatenate((np.asarray(ram.features(), dtype=np.float32), state, template, geometry))


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
