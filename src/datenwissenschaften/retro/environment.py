from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import stable_retro as retro
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

from datenwissenschaften.retro.paths import RetroArenaPaths
from datenwissenschaften.roms import import_roms

GameProvider = Callable[[], str]
SavestateProvider = Callable[[], str | None]
SavestateSetter = Callable[[str | None], None]
EnvWrapperFactory = Callable[[Any], Any]
_last_environment_wrapper: EnvWrapperFactory | None = None


def get_last_environment_wrapper() -> EnvWrapperFactory | None:
    return _last_environment_wrapper


class SavestateResolver:
    """Strategy for choosing a valid stable-retro state."""

    def __init__(self, default_states: Mapping[str, str]) -> None:
        self.default_states = default_states

    def resolve(self, game: str, requested_state: str | None) -> str | None:
        savestate = requested_state or self.default_states.get(game)
        if not savestate:
            logger.warning("No savestate configured for game {}. Starting without a state.", game)
            return None

        available_state_set = {
            state for state in retro.data.list_states(game) if self._is_valid_state_file(game, state)
        }
        if savestate not in available_state_set:
            logger.warning("Savestate {} is not valid for game {}. Starting without a state.", savestate, game)
            return None
        return savestate

    @staticmethod
    def _is_valid_state_file(game: str, state: str) -> bool:
        state_path = retro.data.get_file_path(game, f"{state}.state")
        if not state_path:
            return False
        try:
            with open(state_path, "rb") as file:
                return file.read(2) == b"\x1f\x8b"
        except OSError:
            return False


class RetroEnvironmentFactory:
    """Factory for one wrapped stable-retro environment."""

    def __init__(
        self,
        *,
        paths: RetroArenaPaths,
        wrappers: Mapping[str, type],
        savestate_resolver: SavestateResolver,
        get_game: GameProvider,
        get_savestate: SavestateProvider,
        set_savestate: SavestateSetter,
        obs_size: tuple[int, int],
        action_repeat: int,
        grayscale: bool = True,
        hybrid_obs: bool = False,
    ) -> None:
        self.paths = paths
        self.wrappers = wrappers
        self.savestate_resolver = savestate_resolver
        self.get_game = get_game
        self.get_savestate = get_savestate
        self.set_savestate = set_savestate
        self.obs_size = obs_size
        self.action_repeat = action_repeat
        self.grayscale = grayscale
        self.hybrid_obs = hybrid_obs

    def create(self, number: int):
        record_dir = self.paths.record_dir / str(number)
        record_dir.mkdir(parents=True, exist_ok=True)

        game = self.get_game()
        savestate = self.savestate_resolver.resolve(game, self.get_savestate())
        selected_state_path = retro.data.get_file_path(game, f"{savestate}.state") if savestate else None
        logger.info(
            "make_env selected game={} savestate={} state_path={}",
            game,
            savestate,
            selected_state_path,
        )

        if self.get_savestate() is None:
            self.set_savestate(savestate)

        wrapper_cls = self.wrappers.get(game)
        if wrapper_cls is None:
            raise ValueError(f"Unsupported game: {game}")

        (self.paths.working_dir / game).mkdir(parents=True, exist_ok=True)

        env = retro.make(
            game,
            savestate,
            render_mode="rgb_array",
            record=str(record_dir),
        )

        return wrapper_cls(
            env,
            obs_size=self.obs_size,
            grayscale=self.grayscale,
            hybrid_obs=self.hybrid_obs,
            action_repeat=self.action_repeat,
        )


class EnvironmentBuilder:
    """Builder for a single wrapped stable-retro DummyVecEnv."""

    def __init__(
        self,
        wrapper: EnvWrapperFactory,
        *,
        render_mode: str = "rgb_array",
        n_stack: int = 1,
    ) -> None:
        global _last_environment_wrapper
        self.game = self._required_env("RETRO_ARENA_GAME_ID")
        self.state = os.environ.get("RETRO_ARENA_SAVESTATE")
        self.record_dir = self._required_env("RETRO_ARENA_RECORDING_DIR")
        self.wrapper = wrapper
        _last_environment_wrapper = wrapper
        self.render_mode = render_mode
        self.n_stack = n_stack

    def make_env(self):
        record_dir = os.path.join(self.record_dir, "0")
        os.makedirs(record_dir, exist_ok=True)
        env = retro.make(self.game, self.state, render_mode=self.render_mode, record=record_dir)
        return self.wrapper(env)

    def build(self):
        import_roms()
        venv = DummyVecEnv([self.make_env])
        venv = VecMonitor(venv)
        return VecFrameStack(venv, n_stack=self.n_stack)

    @staticmethod
    def _required_env(name: str) -> str:
        value = os.environ.get(name)
        if value is None:
            raise RuntimeError(f"{name} must be set.")
        return value


class RetroVecEnvBuilder:
    def __init__(self, env_factory: RetroEnvironmentFactory) -> None:
        self.env_factory = env_factory

    def build(self, num_envs: int, n_stack: int):
        venv = SubprocVecEnv([lambda index=index: self.env_factory.create(index) for index in range(num_envs)])
        venv = VecMonitor(venv)
        return VecFrameStack(venv, n_stack=n_stack)
