from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType
from loguru import logger

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target_memory import TargetMemory
from datenwissenschaften.ui import publish_metadata

T = TypeVar("T", bound=RamInfo)


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    training_state_classes: tuple[type[State[T]], ...] = ()
    ram_info_cls: type[T]
    action_repeat = 1
    grayscale = False
    terminate_on_transition = False
    transition_bonus = 0.0

    def __init__(
        self,
        env,
        *,
        obs_size: tuple[int, int],
        action_table: np.ndarray | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        super().__init__(env)

        config = load_config(config_path)
        setup_logging(config.log_level)

        self.state_machine = StateMachine[T](
            self.start_state_cls(),
            terminate_on_transition=self.terminate_on_transition,
            transition_bonus=self.transition_bonus,
            on_transition=self._handle_state_transition,
        )

        if action_table is not None:
            if len(action_table) == 0:
                raise ValueError("Action table must not be empty")

        self.action_space = gym.spaces.Discrete(len(action_table)) if action_table is not None else env.action_space
        self.action_table = action_table
        self.obs_size = obs_size
        self.initial_savestate = config.training.active_savestate
        self._curriculum_root = config.paths.cache_dir / "automatic_savestates" / config.training.game_identity
        self.curriculum = self._create_curriculum()

        self.last_ram: T | None = None
        self.last_frame: np.ndarray | None = None
        self.last_observation: np.ndarray | None = None
        self._started_from_initial_savestate = True
        self._episode_start_state = self.initial_savestate or self.start_state_cls.__name__
        self._episode_bk2_path: str | None = None
        self._state_return = 0.0
        self._state_steps = 0
        self._curriculum_start_state = self.start_state_cls.__name__
        self._curriculum_outcome_recorded = False
        self._curriculum_episode_steps = 0

        channels = 1 if self.grayscale else 3

        visual_space = gym.spaces.Box(low=0, high=255, shape=(channels, *self.obs_size), dtype=np.uint8)
        ram_size = sum(length for _, length in self.ram_info_cls.ram_map().values())
        observation_spaces: dict[str, gym.Space] = {
            "visual": visual_space,
            "ram": gym.spaces.Box(low=0.0, high=1.0, shape=(ram_size,), dtype=np.float32),
        }
        state_classes = dict.fromkeys((self.start_state_cls, *self.training_state_classes))
        self.auxiliary_feature_count = max(
            (len(state_cls().auxiliary_features()) for state_cls in state_classes),
            default=0,
        )
        if self.auxiliary_feature_count:
            observation_spaces["auxiliary"] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.auxiliary_feature_count,),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(observation_spaces)
        self._publish_curriculum_progress()

    def reset(self, **kwargs):
        # Artifact cleanup can remove the recording tree after Retro has been
        # constructed. Retro retains ``movie_path`` but does not recreate its
        # parent directory when it opens the next BK2.
        self._ensure_movie_directory()
        frame, _ = self.env.reset(**kwargs)
        self._episode_bk2_path = self._active_movie_path()

        active_state = self.curriculum.active_state()
        episode_start_state = self.curriculum.episode_start_state()
        restore_checkpoint = episode_start_state is not None
        self._started_from_initial_savestate = not restore_checkpoint
        self._curriculum_start_state = active_state or self.start_state_cls.__name__
        self._curriculum_outcome_recorded = active_state is None
        self._curriculum_episode_steps = 0
        self._episode_start_state = (
            episode_start_state if restore_checkpoint else self.initial_savestate or self.start_state_cls.__name__
        )
        state_cls = self._resolve_state_class(episode_start_state) if restore_checkpoint else None
        if restore_checkpoint:
            frame = self._restore_automatic_savestate(self.curriculum.checkpoint(episode_start_state))

        ram = self._read_ram()
        observation = self._process_observation(frame)

        self.last_ram = ram
        self.last_frame = frame
        self.last_observation = observation

        self.state_machine.reset(ram, frame, observation, state_cls)
        self._state_return = 0.0
        self._state_steps = 0

        return self._agent_observation(observation, ram), {}

    def step(self, action: WrapperActType):
        frame = None
        observation = None
        reward = 0.0
        terminated = False
        truncated = False

        action = self.translate_action(action)
        reward_state = self.state_machine.state_name
        transition: tuple[str, str] | None = None
        curriculum_succeeded = False
        curriculum_mastered = False

        for _ in range(self.action_repeat):
            self._curriculum_episode_steps += 1
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
            reward += state_reward
            terminated = env_terminated or state_terminated
            truncated = env_truncated or state_truncated
            transition = self.state_machine.last_transition or transition

            # Return control immediately so the policy for the new state chooses
            # the very next emulator action during this same episode.
            if self.state_machine.last_transition is not None or terminated or truncated:
                if self.state_machine.last_transition is not None:
                    curriculum_succeeded, curriculum_mastered = self._handle_curriculum_transition(
                        *self.state_machine.last_transition
                    )
                    if curriculum_succeeded:
                        terminated = True
                break

        if frame is None or observation is None:
            raise RuntimeError("No observation produced during step().")

        current_state = self.state_machine.current_state
        won = current_state._won()
        if won:
            curriculum_succeeded = curriculum_succeeded or not self._curriculum_outcome_recorded
            curriculum_mastered = self._record_curriculum_success() or curriculum_mastered
            terminated = True
        elif (terminated or truncated) and not self._curriculum_outcome_recorded:
            checkpoint_deleted = self.curriculum.record_failure(
                self._curriculum_start_state,
                self._curriculum_episode_steps,
                self._state_return + reward,
            )
            if checkpoint_deleted:
                logger.warning(f"Deleted score-stagnant automatic checkpoint for {self._curriculum_start_state}")
            self._publish_curriculum_progress()

        self._state_return += reward
        self._state_steps += 1
        state_return = self._state_return
        state_steps = self._state_steps
        state_segment_end = transition is not None or terminated or truncated
        if transition is not None:
            self._state_return = 0.0
            self._state_steps = 0

        return (
            self._agent_observation(observation, ram),
            reward,
            terminated,
            truncated,
            {
                "won": won,
                "state": self.state_machine.state_name,
                "state_transition": transition,
                "reward_state": reward_state,
                "state_return": state_return,
                "state_steps": state_steps,
                "state_segment_end": state_segment_end,
                "ram": ram.to_dict(),
                "started_from_initial_savestate": self._started_from_initial_savestate,
                "episode_start_state": self._episode_start_state,
                "episode_bk2_path": self._episode_bk2_path,
                "curriculum_state": self._curriculum_start_state,
                "curriculum_succeeded": curriculum_succeeded,
                "curriculum_mastered": curriculum_mastered,
                "curriculum_complete": self.curriculum.is_complete(),
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
        if len(auxiliary_features) > self.auxiliary_feature_count:
            raise RuntimeError(
                f"{self.state_machine.state_name} produced {len(auxiliary_features)} auxiliary features; "
                f"the configured state maximum is {self.auxiliary_feature_count}"
            )
        if self.auxiliary_feature_count:
            padded_features = auxiliary_features + [0.0] * (self.auxiliary_feature_count - len(auxiliary_features))
            agent_observation["auxiliary"] = np.asarray(padded_features, dtype=np.float32)
        return agent_observation

    def state_name(self) -> str:
        return self.state_machine.state_name

    def set_terminate_on_transition(self, enabled: bool) -> bool:
        self.state_machine.terminate_on_transition = bool(enabled)
        return self.state_machine.terminate_on_transition

    def set_transition_bonus(self, bonus: float) -> float:
        self.state_machine.transition_bonus = float(bonus)
        return self.state_machine.transition_bonus

    def set_initial_savestate(self, savestate: str) -> str:
        """Use ``savestate`` for this and subsequent environment resets."""
        self.env.unwrapped.load_state(savestate)
        self.initial_savestate = savestate
        self.curriculum = self._create_curriculum()
        self._publish_curriculum_progress()
        return savestate

    def curriculum_progress(self) -> dict[str, dict[str, int | float | bool | None]]:
        return self.curriculum.progress()

    def reset_training_memory(self) -> None:
        TargetMemory.reset_all()
        self.curriculum = self._create_curriculum()
        self._ensure_movie_directory()
        self._publish_curriculum_progress()

    def _resolve_state_class(self, state_name: str) -> type[State[T]]:
        known_classes = {*self._training_classes(), self.start_state_cls}
        for state_cls in known_classes:
            if state_cls.__name__ == state_name:
                return state_cls
        raise ValueError(f"Unknown training state: {state_name}")

    def _handle_state_transition(self, previous_state_name: str, new_state_name: str) -> None:
        # The state machine changes immediately; the PolicyManager uses the reported
        # state name to select the corresponding model without resetting the level.
        pass

    def _create_curriculum(self) -> ReverseCurriculum:
        scope = self.initial_savestate or "default"
        curriculum = ReverseCurriculum(
            self._curriculum_root / scope,
            [state_cls.__name__ for state_cls in self._training_classes()],
        )
        return curriculum

    def _handle_curriculum_transition(self, previous_state_name: str, new_state_name: str) -> tuple[bool, bool]:
        emulator_state = bytes(self.env.unwrapped.em.get_state())
        if self.curriculum.save_checkpoint(new_state_name, emulator_state):
            logger.info(f"Saved automatic curriculum checkpoint for {new_state_name}")

        # When a rejected checkpoint must be rebuilt, the episode starts at the
        # beginning and traverses the already-mastered prefix. Count steps only
        # after it reaches the current curriculum target.
        if new_state_name == self._curriculum_start_state:
            self._curriculum_episode_steps = 0

        if self._curriculum_outcome_recorded:
            self._publish_curriculum_progress()
            return False, False
        if previous_state_name == self._curriculum_start_state:
            return True, self._record_curriculum_success()
        self._publish_curriculum_progress()
        return False, False

    def _record_curriculum_success(self) -> bool:
        if self._curriculum_outcome_recorded:
            return False
        self._curriculum_outcome_recorded = True
        mastered = self.curriculum.record_success(
            self._curriculum_start_state,
            self._curriculum_episode_steps,
        )
        wins = self.curriculum.wins(self._curriculum_start_state)
        target = self.curriculum.win_target(self._curriculum_start_state)
        if mastered:
            logger.info(f"Mastered curriculum state {self._curriculum_start_state}; advancing to the next state")
        else:
            logger.info(f"Curriculum win for {self._curriculum_start_state}: {wins}/{target} total wins")
        self._publish_curriculum_progress()
        return mastered

    def _restore_automatic_savestate(self, savestate: bytes) -> np.ndarray:
        emulator = self.env.unwrapped
        emulator.em.set_state(savestate)
        movie = getattr(emulator, "movie", None)
        if movie is not None:
            # Retro starts recording during env.reset(), before the curriculum
            # wrapper replaces the configured initial state. Keep the BK2's
            # embedded state aligned with the state used by this episode.
            movie.set_state(savestate)
        emulator.data.reset()
        emulator.data.update_ram()
        return emulator.get_screen(apply_rotation=True)

    def _active_movie_path(self) -> str | None:
        """Return the exact BK2 opened by Stable Retro during the latest reset."""
        emulator = self.env.unwrapped
        movie_path = getattr(emulator, "movie_path", None)
        movie_id = getattr(emulator, "movie_id", None)
        game = getattr(emulator, "gamename", None)
        state = getattr(emulator, "statename", None)
        if not movie_path or not game or not state or not isinstance(movie_id, int) or movie_id < 1:
            return None
        state_name = Path(str(state)).stem
        return str(Path(movie_path) / f"{game}-{state_name}-{movie_id - 1:06d}.bk2")

    def _ensure_movie_directory(self) -> None:
        """Restore Retro's configured recording directory if cleanup removed it."""
        movie_path = getattr(self.env.unwrapped, "movie_path", None)
        if movie_path:
            Path(movie_path).mkdir(parents=True, exist_ok=True)

    def _publish_curriculum_progress(self) -> None:
        publish_metadata("savestate_curriculum", self.curriculum.progress(), replace=True)

    def episode_start_state(self) -> str:
        return self._episode_start_state

    def training_state_names(self) -> list[str]:
        return [state_cls.__name__ for state_cls in self._training_classes()]

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

    def _training_classes(self) -> tuple[type[State[T]], ...]:
        state_classes = self.training_state_classes or (self.start_state_cls,)
        return tuple(state_classes)
