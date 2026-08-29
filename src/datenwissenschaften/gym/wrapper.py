from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.gym.player_motion import PlayerMotion
from datenwissenschaften.ram.model import REQUIRED_RAM_FIELDS, RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.ram_scorer import RamScorerState
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState
from datenwissenschaften.training.episode_counter import EpisodeCounter

T = TypeVar("T", bound=RamInfo)
FRAME_COST = -0.01
STATE_REWARD_LIMIT = 1.0
Observation = np.ndarray


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
        action_size: int,
        model_dir: Path,
        curriculum_enabled: bool,
        state_time_limit_frames: int,
    ) -> None:
        super().__init__(env)

        if self.action_repeat < 1:
            raise ValueError("action_repeat must be positive")

        if action_size < 1:
            raise ValueError("action_size must be positive")
        if state_time_limit_frames < 1:
            raise ValueError("state_time_limit_frames must be positive")

        self.machine = StateMachine(self.start_state_cls(model_dir))
        self.state_types: tuple[type[State[T]], ...] = _state_types(
            self.start_state_cls,
            self.training_state_classes,
        )
        self.episode_counter: EpisodeCounter = EpisodeCounter(model_dir / "episodes.count")
        self.episode_number: int = 0
        self.player_motion = PlayerMotion()
        self.curriculum_enabled = curriculum_enabled
        self.state_time_limit_frames = state_time_limit_frames
        self.state_frames = 0
        self.action_space = gym.spaces.MultiBinary(action_size)
        self.action_space.dtype = np.dtype(np.float32)
        self.previous_action = np.zeros(action_size, dtype=np.float32)
        self.observation_space = _observation_space(
            self.ram_info_cls,
            action_size,
        )
        self.curriculum = ReverseCurriculum(
            model_dir / "curriculum" / self._savestate_name(),
            tuple(state_type.__name__ for state_type in self.state_types),
        )
        self.curriculum_state = self.curriculum.active_state() or self.machine.name
        self.curriculum_steps = 0
        self.curriculum_return = 0.0
        self.curriculum_recorded = False
        self.full_run = False

    def _savestate_name(self) -> str:
        return Path(str(getattr(self.env.unwrapped, "statename", "default"))).stem

    def reset(
        self,
        **kwargs: Any,
    ) -> tuple[Observation, dict[str, Any]]:
        return _reset(self, kwargs)

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        return _step(self, action)


def _reset(
    wrapper: StateMachineGymWrapper[T],
    kwargs: dict[str, Any],
) -> tuple[Observation, dict[str, Any]]:
    frame, info = wrapper.env.reset(**kwargs)
    checkpoint_state = wrapper.curriculum.episode_start_state() if wrapper.curriculum_enabled else None
    if checkpoint_state is not None:
        frame = _restore_checkpoint(wrapper, wrapper.curriculum.checkpoint(checkpoint_state))
    wrapper.episode_number = wrapper.episode_counter.next_episode()
    wrapper.curriculum_state = (
        wrapper.curriculum.active_state() if wrapper.curriculum_enabled else None
    ) or wrapper.state_types[0].__name__
    wrapper.curriculum_steps = 0
    wrapper.state_frames = 0
    wrapper.curriculum_return = (
        wrapper.curriculum.checkpoint_score(checkpoint_state) if checkpoint_state is not None else 0.0
    )
    wrapper.curriculum_recorded = wrapper.curriculum.is_complete() if wrapper.curriculum_enabled else False
    wrapper.full_run = wrapper.curriculum.is_complete() if wrapper.curriculum_enabled else False
    ram = _ram(
        wrapper.ram_info_cls,
        wrapper.env.unwrapped,
    )
    state_type = (
        next(state for state in wrapper.state_types if state.__name__ == checkpoint_state)
        if checkpoint_state is not None
        else wrapper.state_types[0]
    )
    wrapper.machine.reset(
        ram,
        frame,
        state_type,
    )
    velocity = wrapper.player_motion.reset(
        ram,
        frame,
    )
    wrapper.previous_action.fill(0.0)
    return (
        _observation(
            ram,
            wrapper.machine.current,
            velocity,
            wrapper.previous_action,
        ),
        info,
    )


def _step(
    wrapper: StateMachineGymWrapper[T],
    action: WrapperActType,
) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
    reward = 0.0
    controller_action = _controller_action(action, wrapper.action_space)

    for _ in range(wrapper.action_repeat):
        if wrapper.curriculum_enabled:
            wrapper.curriculum_steps += 1
        frame, _, terminated, truncated, info = wrapper.env.step(controller_action)
        ram = _ram(
            wrapper.ram_info_cls,
            wrapper.env.unwrapped,
        )
        previous_state = wrapper.machine.name
        wrapper.state_frames += 1
        state_reward, state_terminated, state_truncated = wrapper.machine.step(
            ram,
            frame,
        )

        transitioned = wrapper.machine.name != previous_state
        if transitioned:
            wrapper.state_frames = 0
        if wrapper.curriculum_enabled and wrapper.full_run:
            wrapper.curriculum_state = wrapper.machine.name
        elif wrapper.state_frames >= wrapper.state_time_limit_frames:
            state_truncated = True
        terminated = terminated or state_terminated
        truncated = truncated or state_truncated
        reward += _limit_automatic_reward(state_reward) + FRAME_COST
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
    if wrapper.curriculum_enabled:
        wrapper.curriculum_return += reward
        if transitioned:
            _save_curriculum_checkpoint(wrapper)
            if previous_state == wrapper.curriculum_state and not wrapper.curriculum_recorded:
                wrapper.curriculum_recorded = True
                wrapper.curriculum.record_success(wrapper.curriculum_state, wrapper.curriculum_steps)
                terminated = True
        elif (terminated or truncated) and not wrapper.curriculum_recorded:
            wrapper.curriculum.record_failure(
                wrapper.curriculum_state,
                wrapper.curriculum_steps,
                wrapper.curriculum_return,
            )
        if won and not wrapper.curriculum_recorded:
            wrapper.curriculum_recorded = True
            wrapper.curriculum.record_success(wrapper.curriculum_state, wrapper.curriculum_steps)
    info.update(_episode_info(wrapper, ram, won))
    if wrapper.curriculum_enabled:
        info["full_run"] = wrapper.full_run
        info["curriculum_state"] = wrapper.curriculum_state
        info["curriculum_complete"] = wrapper.curriculum.is_complete()
        info["curriculum_progress"] = wrapper.curriculum.progress()

    if terminated or truncated:
        recording_path = _recording_path(wrapper.env.unwrapped)
        if recording_path is not None:
            info["episode_bk2_path"] = recording_path

    velocity = wrapper.player_motion.measure(
        ram,
        frame,
    )
    wrapper.previous_action = controller_action.astype(np.float32)
    observation = _observation(
        ram,
        wrapper.machine.current,
        velocity,
        wrapper.previous_action,
    )
    return observation, reward, terminated, truncated, info


def _limit_automatic_reward(reward: float) -> float:
    if not np.isfinite(reward):
        raise ValueError(f"Automatic reward must be finite, got {reward}")
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


def _recording_path(retro: Any) -> str | None:
    if retro.movie_path is None:
        return None
    recording = Path(retro.movie_path) / (f"{retro.gamename}-{Path(retro.statename).stem}-{retro.movie_id - 1:06d}.bk2")
    return str(recording)


def _save_curriculum_checkpoint(wrapper: StateMachineGymWrapper[Any]) -> None:
    emulator = wrapper.env.unwrapped
    wrapper.curriculum.save_checkpoint(
        wrapper.machine.name,
        bytes(emulator.em.get_state()),
        wrapper.curriculum_return,
    )


def _restore_checkpoint(wrapper: StateMachineGymWrapper[Any], state: bytes) -> np.ndarray:
    emulator = wrapper.env.unwrapped
    emulator.em.set_state(state)
    emulator.data.reset()
    emulator.data.update_ram()
    return emulator.get_screen(apply_rotation=True)


def _observation_space(
    ram_info: type[RamInfo],
    action_count: int,
) -> gym.spaces.Box:
    ram_map = ram_info.ram_map()
    missing = tuple(field for field in REQUIRED_RAM_FIELDS if field not in ram_map)

    if missing:
        fields = ", ".join(f"ram.{field}" for field in missing)
        raise ValueError(f"State policy requires RAM fields: {fields}")

    ram_size = ram_info.feature_size()

    return gym.spaces.Box(
        -1.0,
        1.0,
        shape=(ram_size + action_count + 5,),
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
    current: State[T],
    velocity: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    target_features = np.zeros(3, dtype=np.float32)
    if isinstance(current, TargetState):
        target_features = current.target_features()
    elif isinstance(current, RamScorerState):
        target_features = current.target_features()

    if not isinstance(current, (TargetState, RamScorerState)):
        raise TypeError(f"Unsupported state type: {type(current).__name__}")

    return np.concatenate(
        (
            np.asarray(ram.features(), dtype=np.float32),
            target_features,
            velocity,
            previous_action,
        ),
    )


def _controller_action(
    action: WrapperActType,
    action_space: gym.Space,
) -> np.ndarray:
    controller_action = np.asarray(action, dtype=np.int8)
    if not action_space.contains(controller_action):
        raise ValueError(f"Action {controller_action} is outside {action_space}")
    return controller_action
