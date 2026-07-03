import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType
from loguru import logger

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    training_state_classes: tuple[type[State[T]], ...] = ()
    ram_info_cls: type[T]
    action_repeat = 1
    grayscale = False

    def __init__(
        self,
        env,
        *,
        obs_size: tuple[int, int],
        action_table: np.ndarray | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        **_legacy_options,
    ):
        super().__init__(env)

        self.state_machine = StateMachine[T](self.start_state_cls())

        if action_table is not None:
            if len(action_table) == 0:
                raise ValueError("Action table must not be empty")

        self.action_space = gym.spaces.Discrete(len(action_table)) if action_table is not None else env.action_space
        self.action_table = action_table
        self.obs_size = obs_size
        config = load_config(config_path)
        self.savestate_dir = config.paths.savestate_dir
        self.savestate_beaten_threshold = config.training.savestate_beaten_threshold

        self.last_ram: T | None = None
        self.last_frame: np.ndarray | None = None
        self.last_observation: np.ndarray | None = None
        self._started_from_initial_savestate = True

        self._last_progress: float | int | None = None
        self._frames_without_progress = 0

        self._validate_training_states()
        self._load_savestates()

        channels = 1 if self.grayscale else 3

        visual_space = gym.spaces.Box(low=0, high=255, shape=(channels, *self.obs_size), dtype=np.uint8)
        ram_size = sum(length for _, length in self.ram_info_cls.ram_map().values())
        observation_spaces: dict[str, gym.Space] = {
            "visual": visual_space,
            "ram": gym.spaces.Box(low=0.0, high=1.0, shape=(ram_size,), dtype=np.float32),
        }
        auxiliary_features = self.state_machine.current_state.auxiliary_features()
        self.auxiliary_feature_count = len(auxiliary_features)
        if self.auxiliary_feature_count:
            observation_spaces["auxiliary"] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.auxiliary_feature_count,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(observation_spaces)

    def reset(self, **kwargs):
        frame, _ = self.env.reset(**kwargs)

        self._last_progress = None
        self._frames_without_progress = 0

        self._load_savestates()
        state_cls = self._highest_savestate_state()
        self._started_from_initial_savestate = state_cls is None
        if state_cls is not None:
            frame = self._restore_savestate(self.state_machine.savestate(state_cls))

        ram = self._read_ram()
        observation = self._process_observation(frame)

        self.last_ram = ram
        self.last_frame = frame
        self.last_observation = observation

        self.state_machine.reset(ram, frame, observation, state_cls)
        self._update_progress_tracking()

        return self._agent_observation(observation, ram), {}

    def step(self, action: WrapperActType):
        frame = None
        observation = None
        reward = 0.0
        terminated = False
        truncated = False

        action = self.translate_action(action)

        for _ in range(self.action_repeat):
            state_before_step = type(self.state_machine.current_state)

            frame, _, env_terminated, env_truncated, _ = self.env.step(action)

            ram = self._read_ram()
            observation = self._process_observation(frame)

            self.last_ram = ram
            self.last_frame = frame
            self.last_observation = observation

            state_reward, state_terminated, state_truncated = self.state_machine.step(
                ram,
                frame,
                observation,
            )

            self._update_progress_tracking()

            state_after_step = type(self.state_machine.current_state)
            if state_after_step.progress > state_before_step.progress:
                if self._mark_beaten(state_before_step):
                    self._load_savestate(state_after_step)

                    savestate = self.env.unwrapped.em.get_state()
                    if self.state_machine.save_current_state(savestate):
                        self._write_savestate(state_after_step.__name__, savestate)
                        logger.info(f"Saved automatic savestate for {state_after_step.__name__}")

            reward += state_reward
            terminated = env_terminated or state_terminated
            truncated = env_truncated or state_truncated

            if terminated or truncated:
                break

        if frame is None or observation is None:
            raise RuntimeError("No observation produced during step().")

        current_state = self.state_machine.current_state
        won = current_state._won()
        if won:
            self._mark_beaten(type(current_state))

        return (
            self._agent_observation(observation, ram),
            reward,
            terminated,
            truncated,
            {
                "won": won,
                "state": self.state_machine.state_name,
                "ram": ram.to_dict(),
                "progress": type(current_state).progress,
                "frames_without_progress": self._frames_without_progress,
                "started_from_initial_savestate": self._started_from_initial_savestate,
            },
        )

    def features(self) -> list[float]:
        return self.state_machine.features()

    def policy_input(self) -> tuple[np.ndarray, str]:
        features = np.asarray(self.state_machine.features(), dtype=np.float32)
        return features, self.state_machine.state_name

    def _agent_observation(self, observation: np.ndarray, ram: RamInfo) -> dict[str, np.ndarray]:
        agent_observation = {
            "visual": observation,
            "ram": np.asarray(ram.features(), dtype=np.float32),
        }
        auxiliary_features = self.state_machine.current_state.auxiliary_features(ram)
        if len(auxiliary_features) != self.auxiliary_feature_count:
            raise RuntimeError(
                f"{self.state_machine.state_name} produced {len(auxiliary_features)} auxiliary features; "
                f"expected {self.auxiliary_feature_count}"
            )
        if auxiliary_features:
            agent_observation["auxiliary"] = np.asarray(auxiliary_features, dtype=np.float32)
        return agent_observation

    def state_name(self) -> str:
        return self.state_machine.state_name

    def training_state_names(self) -> list[str]:
        return [state_cls.__name__ for state_cls in self._training_classes()]

    def set_training_state(self, state_name: str) -> bool:
        classes_by_name = self._training_classes_by_name()
        if state_name not in classes_by_name:
            raise ValueError(f"Unknown training state: {state_name}")

        self._load_savestates()
        selected = self._highest_savestate_state()
        return selected is not None and selected.__name__ == state_name

    def highest_progress_state(self) -> str:
        self._load_savestates()
        state_cls = self._highest_savestate_state()
        return state_cls.__name__ if state_cls is not None else self.start_state_cls.__name__

    def active_savestate_state(self) -> str | None:
        self._load_savestates()
        state_cls = self._highest_savestate_state()
        return state_cls.__name__ if state_cls is not None else None

    def frames_without_progress(self) -> int:
        return self._frames_without_progress

    def delete_savestate(self, state_name: str) -> bool:
        classes_by_name = self._training_classes_by_name()
        try:
            state_cls = classes_by_name[state_name]
        except KeyError as error:
            raise ValueError(f"Unknown training state: {state_name}") from error

        deleted = self.state_machine.delete_savestate(state_cls)
        return self._delete_savestate_file(state_name) or deleted

    def clear_training_progress(self) -> None:
        for state_cls in self._training_classes():
            self.state_machine.clear_saved_progress(state_cls)

    def translate_action(self, action: WrapperActType):
        if self.action_table is None:
            return action

        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise TypeError(f"Action must be an integer, got {type(action).__name__}")

        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(f"Action {action_index} is outside {self.action_space}")

        if action_index >= len(self.action_table):
            raise ValueError(f"Action {action_index} has no entry in the action table")

        return self.action_table[action_index]

    def _read_ram(self) -> T:
        return self.ram_info_cls.from_ram(self.env.unwrapped.get_ram())

    def _process_observation(self, obs: Any) -> np.ndarray:
        if self.grayscale:
            if cv2 is not None:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
                obs = cv2.resize(
                    obs,
                    (self.obs_size[1], self.obs_size[0]),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                obs = obs.mean(axis=2).astype(np.uint8)
                obs = self._resize_nearest(obs)

            return np.expand_dims(obs, axis=0)

        if cv2 is not None:
            obs = cv2.resize(
                obs,
                (self.obs_size[1], self.obs_size[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            obs = self._resize_nearest(obs)

        return np.transpose(obs, (2, 0, 1))

    def _resize_nearest(self, obs: np.ndarray) -> np.ndarray:
        target_h, target_w = self.obs_size
        y_index = np.linspace(0, obs.shape[0] - 1, target_h).astype(int)
        x_index = np.linspace(0, obs.shape[1] - 1, target_w).astype(int)
        return obs[y_index][:, x_index]

    def num_actions(self) -> int:
        return self.action_space.n

    def _update_progress_tracking(self) -> None:
        progress = type(self.state_machine.current_state).progress

        if self._last_progress is None:
            self._last_progress = progress
            self._frames_without_progress = 0
            return

        if progress > self._last_progress:
            self._last_progress = progress
            self._frames_without_progress = 0
            return

        self._frames_without_progress += 1

    def _highest_savestate_state(self) -> type[State[T]] | None:
        available = [
            state_cls
            for state_cls in self._training_classes()
            if not self.state_machine.is_beaten(state_cls) and self.state_machine.savestate(state_cls) is not None
        ]
        return max(available, key=lambda state_cls: state_cls.progress, default=None)

    def _restore_savestate(self, savestate: bytes | None) -> np.ndarray:
        if savestate is None:
            raise RuntimeError("Cannot restore an empty savestate.")

        emulator = self.env.unwrapped
        emulator.em.set_state(savestate)
        emulator.data.reset()
        emulator.data.update_ram()
        return emulator.get_screen(apply_rotation=True)

    def _training_classes_by_name(self) -> dict[str, type[State[T]]]:
        return {state_cls.__name__: state_cls for state_cls in self._training_classes()}

    def _training_classes(self) -> tuple[type[State[T]], ...]:
        state_classes = self.training_state_classes or (self.start_state_cls,)
        return tuple(sorted(state_classes, key=lambda state_cls: state_cls.progress))

    def _validate_training_states(self) -> None:
        state_classes = self.training_state_classes or (self.start_state_cls,)
        progresses = [state_cls.progress for state_cls in state_classes]
        if len(progresses) != len(set(progresses)):
            raise ValueError(f"Training state progress values must be unique: {progresses}")

    def _load_savestates(self) -> None:
        for state_cls in self._training_classes_by_name().values():
            self._load_savestate(state_cls)

    def _load_savestate(self, state_cls: type[State[T]]) -> bool:
        if self._beaten_count(state_cls.__name__) >= self.savestate_beaten_threshold:
            self.state_machine.mark_beaten(state_cls)
            self._delete_savestate_file(state_cls.__name__)
            return False

        path = self._savestate_path(state_cls.__name__)
        if not path.is_file():
            return False

        savestate = path.read_bytes()
        if not savestate:
            return False

        self.state_machine.load_savestate(state_cls, savestate)
        return True

    def _write_savestate(self, state_name: str, savestate: bytes) -> None:
        path = self._savestate_path(state_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary_path.write_bytes(bytes(savestate))
        temporary_path.replace(path)

    def _savestate_path(self, state_name: str) -> Path:
        return self.savestate_dir / f"{state_name}.state"

    def _beaten_path(self, state_name: str) -> Path:
        return self.savestate_dir / f"{state_name}.beaten"

    def _beaten_lock_path(self, state_name: str) -> Path:
        return self.savestate_dir / f".{state_name}.beaten.lock"

    def _mark_beaten(self, state_cls: type[State[T]]) -> bool:
        state_name = state_cls.__name__
        with self._beaten_lock(state_name):
            beaten_count = min(self.savestate_beaten_threshold, self._beaten_count(state_name) + 1)
            self._write_beaten_count(state_name, beaten_count)
            if beaten_count < self.savestate_beaten_threshold:
                logger.debug(
                    f"Recorded savestate victory for {state_name} "
                    f"({beaten_count}/{self.savestate_beaten_threshold})"
                )
                return False

            self.state_machine.mark_beaten(state_cls)
            self._delete_savestate_file(state_name)
            logger.debug(
                f"Marked savestate as beaten for {state_name} " f"({beaten_count}/{self.savestate_beaten_threshold})"
            )
            return True

    def _beaten_count(self, state_name: str) -> int:
        path = self._beaten_path(state_name)
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return 0

        if not value:
            return self.savestate_beaten_threshold
        try:
            return max(0, int(value))
        except ValueError:
            logger.warning(f"Invalid beaten count in {path}; treating the savestate as beaten")
            return self.savestate_beaten_threshold

    def _write_beaten_count(self, state_name: str, beaten_count: int) -> None:
        path = self._beaten_path(state_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary_path.write_text(str(beaten_count), encoding="utf-8")
        temporary_path.replace(path)

    @contextmanager
    def _beaten_lock(self, state_name: str) -> Iterator[None]:
        path = self._beaten_lock_path(state_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _delete_savestate_file(self, state_name: str) -> bool:
        try:
            self._savestate_path(state_name).unlink()
            return True
        except FileNotFoundError:
            return False
