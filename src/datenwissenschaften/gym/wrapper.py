from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType
from loguru import logger

from datenwissenschaften.curriculum.progress import SavestateCurriculum
from datenwissenschaften.gym.player_motion import PlayerMotion
from datenwissenschaften.ram.model import REQUIRED_DQN_RAM_FIELDS, RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State
from datenwissenschaften.training.episode_counter import EpisodeCounter

T = TypeVar("T", bound=RamInfo)


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    training_state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_repeat: int
    transition_reward: float
    victory_reward: float
    failure_penalty: float
    curriculum_successes: int
    curriculum_stagnation_episodes: int
    full_run_probability: float

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
        self.curriculum = _curriculum(self, model_dir)
        self.episode_counter: EpisodeCounter = EpisodeCounter(model_dir / "episodes.count")
        self.episode_number: int = 0
        self.episode_score = 0.0
        self.initial_episode_state = self.state_types[0].__name__
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
        frame, info = self.env.reset(**kwargs)
        self.episode_number = self.episode_counter.next_episode()
        self.episode_score = 0.0

        state_name, savestate = self.curriculum.start()
        self.initial_episode_state = state_name

        if savestate is not None:
            frame = _restore(
                self.env.unwrapped,
                savestate,
            )

        ram = _ram(
            self.ram_info_cls,
            self.env.unwrapped,
        )
        state_type = next(state for state in self.state_types if state.__name__ == state_name)

        self.machine.reset(
            ram,
            frame,
            state_type,
        )
        velocity = self.player_motion.reset(
            ram,
            frame,
        )

        observation = _observation(
            ram,
            self.state_types,
            self.machine.current,
            velocity,
        )

        return observation, info

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 0.0

        for _ in range(self.action_repeat):
            frame, _, terminated, truncated, info = self.env.step(
                _action(
                    action,
                    self.action_table,
                    self.action_space,
                )
            )

            ram = _ram(
                self.ram_info_cls,
                self.env.unwrapped,
            )
            previous_state = self.machine.name

            state_reward, state_terminated, state_truncated = self.machine.step(
                ram,
                frame,
            )

            if self.machine.name != previous_state:
                state_reward += self.transition_reward
                _record_transition(
                    self.curriculum,
                    previous_state,
                    self.machine.name,
                    bytes(self.env.unwrapped.em.get_state()),
                )

            reward += state_reward
            self.episode_score += state_reward

            terminated = terminated or state_terminated
            truncated = truncated or state_truncated

            if terminated or truncated:
                break

        won = self.machine.current._won()

        if won:
            outcome_reward = self.victory_reward
        elif terminated:
            outcome_reward = self.failure_penalty
        else:
            outcome_reward = 0.0

        reward += outcome_reward
        self.episode_score += outcome_reward

        if won:
            _record_victory(
                self.curriculum,
                self.machine.name,
            )

        terminated = terminated or won

        if (terminated or truncated) and not self.curriculum.recorded:
            diagnostics = self.curriculum.record_attempt(
                self.episode_score,
                self.machine.name != self.initial_episode_state or won,
            )
            if diagnostics.deleted:
                logger.warning(
                    "Deleted bad curriculum savestate {} after {} attempts: "
                    "recent_median={:.3f}, best_median={:.3f}, "
                    "trend={:.6f}, progress_rate={:.1%}, stagnant_windows={}",
                    diagnostics.state,
                    diagnostics.attempts,
                    diagnostics.recent_median,
                    diagnostics.best_median,
                    diagnostics.trend,
                    diagnostics.progress_rate,
                    diagnostics.stagnant_windows,
                )

        info.update(
            {
                "state": self.machine.name,
                "episode_number": self.episode_number,
                "episode_state": self.curriculum.episode_state,
                "full_run": self.curriculum.full_run,
                "action_repeat": self.action_repeat,
                "won": won,
                "ram": ram.to_dict(),
            }
        )

        if terminated or truncated:
            info["episode_bk2_path"] = _recording_path(self.env.unwrapped)

        velocity = self.player_motion.measure(
            ram,
            frame,
        )
        observation = _observation(
            ram,
            self.state_types,
            self.machine.current,
            velocity,
        )

        return observation, reward, terminated, truncated, info


def _curriculum(
    wrapper: "StateMachineGymWrapper[Any]",
    model_dir: Path,
) -> SavestateCurriculum:
    states = tuple(state.__name__ for state in wrapper.state_types)

    return SavestateCurriculum(
        model_dir / "curriculum",
        states,
        wrapper.curriculum_successes,
        wrapper.curriculum_stagnation_episodes,
        wrapper.full_run_probability,
    )


def _record_transition(
    curriculum: SavestateCurriculum,
    previous: str,
    current: str,
    savestate: bytes,
) -> None:
    successes, completed = curriculum.transition(
        previous,
        current,
        savestate,
    )
    _log_mastery(
        curriculum,
        previous,
        successes,
        completed,
    )


def _record_victory(
    curriculum: SavestateCurriculum,
    state: str,
) -> None:
    if curriculum.full_run:
        logger.success("Completed full run")
        return

    successes, completed = curriculum.victory(state)
    _log_mastery(
        curriculum,
        state,
        successes,
        completed,
    )


def _log_mastery(
    curriculum: SavestateCurriculum,
    state: str,
    successes: int,
    completed: bool,
) -> None:
    if completed:
        logger.success(
            "Mastered curriculum state {} after {} successes",
            state,
            successes,
        )
    elif successes:
        logger.info(
            "Curriculum state {} success {}/{}",
            state,
            successes,
            curriculum.required_successes,
        )


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


def _restore(
    emulator: Any,
    savestate: bytes,
) -> np.ndarray:
    emulator.em.set_state(savestate)
    emulator.data.reset()
    emulator.data.update_ram()

    return emulator.get_screen(apply_rotation=True)
