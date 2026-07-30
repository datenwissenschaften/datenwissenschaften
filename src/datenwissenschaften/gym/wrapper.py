from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType

from datenwissenschaften.gym.player_motion import PlayerMotion
from datenwissenschaften.ram.model import REQUIRED_DQN_RAM_FIELDS, RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State
from datenwissenschaften.training.episode_counter import EpisodeCounter

T = TypeVar("T", bound=RamInfo)
FRAME_COST = -0.01
STATE_REWARD_LIMIT = 1.0


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    training_state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_repeat: int
    transition_reward: float
    victory_reward: float
    failure_penalty: float

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

        if action_table is None:
            raise ValueError("Feature-based DQN requires a discrete action table")

        self.machine = StateMachine(self.start_state_cls(model_dir))
        self.state_types: tuple[type[State[T]], ...] = _state_types(
            self.start_state_cls,
            self.training_state_classes,
        )
        self.episode_counter: EpisodeCounter = EpisodeCounter(model_dir / "episodes.count")
        self.episode_number: int = 0
        self.player_motion = PlayerMotion()
        self.action_table = action_table
        self.action_space = gym.spaces.Discrete(len(action_table))
        self.observation_space = _observation_space(
            self.ram_info_cls,
            self.state_types,
        )

    def reset(
        self,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return _reset(self, kwargs)

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return _step(self, action)


def _reset(
    wrapper: StateMachineGymWrapper[T],
    kwargs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    frame, info = wrapper.env.reset(**kwargs)
    wrapper.episode_number = wrapper.episode_counter.next_episode()
    ram = _ram(
        wrapper.ram_info_cls,
        wrapper.env.unwrapped,
    )
    wrapper.machine.reset(
        ram,
        frame,
        wrapper.state_types[0],
    )
    velocity = wrapper.player_motion.reset(
        ram,
        frame,
    )
    return (
        _observation(
            ram,
            wrapper.state_types,
            wrapper.machine.current,
            velocity,
        ),
        info,
    )


def _step(
    wrapper: StateMachineGymWrapper[T],
    action: WrapperActType,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    reward = 0.0

    for _ in range(wrapper.action_repeat):
        frame, _, terminated, truncated, info = wrapper.env.step(
            _action(
                action,
                wrapper.action_table,
                wrapper.action_space,
            )
        )
        ram = _ram(
            wrapper.ram_info_cls,
            wrapper.env.unwrapped,
        )
        previous_state = wrapper.machine.name
        state_reward, state_terminated, state_truncated = wrapper.machine.step(
            ram,
            frame,
        )

        transitioned = wrapper.machine.name != previous_state
        terminated = terminated or state_terminated
        truncated = truncated or state_truncated
        reward += _shape_reward(state_reward) + FRAME_COST
        if transitioned and not terminated and not truncated:
            reward += wrapper.transition_reward

        if transitioned or terminated or truncated:
            break

    won = wrapper.machine.current._won()
    if won:
        outcome_reward = wrapper.victory_reward
    elif terminated or truncated:
        outcome_reward = wrapper.failure_penalty
    else:
        outcome_reward = 0.0

    reward += outcome_reward
    terminated = terminated or won
    info.update(_episode_info(wrapper, ram, won))

    if terminated or truncated:
        info["episode_bk2_path"] = _recording_path(wrapper.env.unwrapped)

    velocity = wrapper.player_motion.measure(
        ram,
        frame,
    )
    observation = _observation(
        ram,
        wrapper.state_types,
        wrapper.machine.current,
        velocity,
    )
    return observation, reward, terminated, truncated, info


def _shape_reward(reward: float) -> float:
    if not np.isfinite(reward):
        raise ValueError(f"State reward must be finite, got {reward}")
    return float(np.clip(reward, -STATE_REWARD_LIMIT, STATE_REWARD_LIMIT))


def _episode_info(
    wrapper: StateMachineGymWrapper[Any],
    ram: RamInfo,
    won: bool,
) -> dict[str, Any]:
    return {
        "state": wrapper.machine.name,
        "episode_number": wrapper.episode_number,
        "action_repeat": wrapper.action_repeat,
        "won": won,
        "ram": ram.to_dict(),
    }


def _recording_path(retro: Any) -> str:
    recording = Path(retro.movie_path) / (f"{retro.gamename}-{Path(retro.statename).stem}-{retro.movie_id - 1:06d}.bk2")
    return str(recording)


def _observation_space(
    ram_info: type[RamInfo],
    states: tuple[type[State[Any]], ...],
) -> gym.spaces.Box:
    ram_map = ram_info.ram_map()
    missing = tuple(field for field in REQUIRED_DQN_RAM_FIELDS if field not in ram_map)

    if missing:
        fields = ", ".join(f"ram.{field}" for field in missing)
        raise ValueError(f"Feature-based DQN requires RAM fields: {fields}")

    ram_size = sum(length for _, length in ram_map.values())

    return gym.spaces.Box(
        -1.0,
        1.0,
        shape=(ram_size + len(states) + 5,),
        dtype=np.float32,
    )


def _ram(
    model: type[T],
    emulator: Any,
) -> T:
    return model.from_ram(emulator.get_ram())


def _state_types(
    start: type[State[T]],
    states: tuple[type[State[T]], ...],
) -> tuple[type[State[T]], ...]:
    return tuple(dict.fromkeys((start, *states)))


def _observation(
    ram: T,
    states: tuple[type[State[T]], ...],
    current: State[T],
    velocity: np.ndarray,
) -> np.ndarray:
    state = np.zeros(
        len(states),
        dtype=np.float32,
    )
    state[states.index(type(current))] = 1.0

    template = np.zeros(
        3,
        dtype=np.float32,
    )

    if current.target_detector is not None:
        template[0] = float(current.target_detector.seen)

        if current.target_detector.position is not None:
            height, width = current.frame.shape[:2]
            template[1] = current.target_detector.position[0] / width
            template[2] = current.target_detector.position[1] / height

    return np.concatenate(
        (
            np.asarray(
                ram.features(),
                dtype=np.float32,
            ),
            state,
            template,
            velocity,
        )
    )


def _action(
    action: WrapperActType,
    action_table: np.ndarray,
    action_space: gym.Space,
) -> WrapperActType:
    if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
        raise TypeError(f"Action must be an integer, got {type(action).__name__}")

    index = int(action)

    if not action_space.contains(index):
        raise ValueError(f"Action {index} is outside {action_space}")

    return action_table[index]
