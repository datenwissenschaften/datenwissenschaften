from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType
from loguru import logger

from datenwissenschaften.curriculum.progress import SavestateCurriculum
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
        self.state_types: tuple[type[State[T]], ...] = _state_types(
            self.start_state_cls, self.training_state_classes
        )
        self.curriculum: SavestateCurriculum = SavestateCurriculum(
            model_dir / "curriculum", tuple(state.__name__ for state in self.state_types)
        )
        self.episode_score: float = 0.0
        self.obs_size = obs_size
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(action_table)) if action_table is not None else env.action_space
        channels = 1 if self.grayscale else 3
        state_count = len(self.state_types)
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
        self.episode_score = 0.0
        state_name, savestate = self.curriculum.start()
        if savestate is not None:
            frame = _restore(self.env.unwrapped, savestate)
        ram = _ram(self.ram_info_cls, self.env.unwrapped)
        observation = _image(frame, self.obs_size, self.grayscale)
        state_type = next(state for state in self.state_types if state.__name__ == state_name)
        self.machine.reset(ram, frame, observation, state_type)
        return _observation(observation, ram, self.state_types, type(self.machine.current)), info

    def step(self, action: WrapperActType) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        reward = 0.0
        for _ in range(self.action_repeat):
            frame, _, terminated, truncated, info = self.env.step(_action(action, self.action_table, self.action_space))
            ram = _ram(self.ram_info_cls, self.env.unwrapped)
            observation = _image(frame, self.obs_size, self.grayscale)
            previous_state = self.machine.name
            state_reward, state_terminated, state_truncated = self.machine.step(ram, frame, observation)
            if (
                self.machine.name != previous_state
                and not self.curriculum.recorded
                and previous_state == self.curriculum.episode_state
            ):
                wins, mastered = self.curriculum.transition(
                    previous_state,
                    self.machine.name,
                    bytes(self.env.unwrapped.em.get_state()),
                )
                _log_progress(previous_state, wins, mastered)
            reward += state_reward
            self.episode_score += state_reward
            terminated = terminated or state_terminated
            truncated = truncated or state_truncated
            if terminated or truncated:
                break
        won = self.machine.current._won()
        if won and not self.curriculum.recorded and self.machine.name == self.curriculum.episode_state:
            wins, mastered = self.curriculum.victory(self.machine.name)
            _log_progress(self.machine.name, wins, mastered)
        terminated = terminated or won
        if (terminated or truncated) and not self.curriculum.recorded:
            attempts, deleted = self.curriculum.record_attempt(self.episode_score)
            _log_stagnation(self.curriculum.episode_state, attempts, deleted)
        info.update({"state": self.machine.name, "won": won, "ram": ram.to_dict()})
        if terminated or truncated:
            retro = self.env.unwrapped
            recording = Path(retro.movie_path) / (
                f"{retro.gamename}-{Path(retro.statename).stem}-{retro.movie_id - 1:06d}.bk2"
            )
            info["episode_bk2_path"] = str(recording)
        agent_observation = _observation(observation, ram, self.state_types, type(self.machine.current))
        return agent_observation, reward, terminated, truncated, info

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
    image: np.ndarray,
    ram: T,
    states: tuple[type[State[T]], ...],
    current: type[State[T]],
) -> dict[str, np.ndarray]:
    active_state = np.zeros(len(states), dtype=np.float32)
    active_state[states.index(current)] = 1.0
    return {"visual": image, "ram": np.asarray(ram.features(), dtype=np.float32), "state": active_state}


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


def _image(frame: np.ndarray, size: tuple[int, int], grayscale: bool) -> np.ndarray:
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(frame, size[::-1], interpolation=cv2.INTER_AREA)
    return resized[None, ...] if grayscale else np.transpose(resized, (2, 0, 1))


def _log_progress(state: str, wins: int, mastered: bool) -> None:
    if mastered:
        logger.success("Mastered curriculum state {} after {} wins", state, wins)
        return
    logger.info("Curriculum state {}: {}/{} wins", state, wins, SavestateCurriculum.wins_required)


def _log_stagnation(state: str, attempts: int, deleted: bool) -> None:
    if deleted:
        logger.warning("Deleted stagnant curriculum savestate {} after {} attempts", state, attempts)
