from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.callbacks import (
    BestEpisodeCallback,
    SaveModelCallback,
    StopTrainingAtTimestepsCallback,
)
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.model import get_model_metadata, get_model_path
from datenwissenschaften.retro.environment import get_last_environment_wrapper
from datenwissenschaften.runtime import RetroSpeedlabRuntime, configure_runtime
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, RetroSpeedlabConfig, load_config
from datenwissenschaften.ui import configure_history, publish_metadata, start_ui
from datenwissenschaften.ui.control import configure_training_control


class Trainer:
    # noinspection PyTypeChecker
    def __init__(
        self,
        *,
        additional_callbacks: Sequence[BaseCallback] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config: RetroSpeedlabConfig = load_config(config_path)
        setup_logging(self.config.log_level)
        self.total_timesteps = self.config.training.total_timesteps
        self._additional_callbacks = list(additional_callbacks or [])
        self.callbacks = self._default_callbacks() + self._additional_callbacks
        self._state: dict[str, Any] = {}
        self._savestate = self.config.training.savestate

    def train(self, model) -> None:
        configure_accelerator()
        self._configure_runtime()
        self._start_ui(model)
        if model.num_timesteps >= self.total_timesteps:
            return

        model.learn(
            total_timesteps=self.total_timesteps - model.num_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
        )

    def _start_ui(self, model) -> None:
        if not self.config.ui.enabled:
            return
        history_dir = self.config.paths.models_dir / self.config.training.game
        if self.config.training.savestate:
            history_dir /= self.config.training.savestate
        configure_history(history_dir / "history.json")
        configure_training_control(
            game=self.config.training.game,
            model_dir=self.config.paths.models_dir / self.config.training.game,
            restart_supported=bool(getattr(model, "supports_ui_restart", False)),
            on_reset=self._reset_for_restart,
        )
        if start_ui(self.config.ui) is None:
            return
        env = model.get_env() if callable(getattr(model, "get_env", None)) else getattr(model, "env", None)
        publish_metadata(
            "run",
            {
                "game": self.config.training.game,
                "savestate": self.config.training.savestate,
                "total_timesteps": self.total_timesteps,
                "population_size": self.config.training.population_size,
                "configured_envs": self.config.training.num_envs,
            },
        )
        publish_metadata("model", get_model_metadata(model))
        publish_metadata("environment", self._environment_metadata(env))

    def _reset_for_restart(self) -> None:
        self._state.clear()
        self._savestate = self.config.training.savestate
        self.callbacks[:] = self._default_callbacks() + self._additional_callbacks

    @staticmethod
    def _environment_metadata(env) -> dict[str, Any]:
        if env is None:
            return {"class": None}
        wrappers = []
        current = env
        while current is not None and len(wrappers) < 12:
            wrappers.append(f"{current.__class__.__module__}.{current.__class__.__qualname__}")
            next_env = getattr(current, "venv", None)
            if next_env is None:
                next_env = getattr(current, "env", None)
            if next_env is current:
                break
            current = next_env
        emulator_action_space = str(getattr(env, "action_space", None))
        action_space = emulator_action_space
        try:
            num_actions = env.env_method("num_actions")[0]
            if isinstance(num_actions, int) and not isinstance(num_actions, bool) and num_actions > 0:
                action_space = f"Discrete({num_actions})"
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return {
            "class": wrappers[0],
            "wrappers": wrappers,
            "num_envs": getattr(env, "num_envs", None),
            "observation_space": str(getattr(env, "observation_space", None)),
            "action_space": action_space,
            "emulator_action_space": emulator_action_space,
        }

    def _default_callbacks(self) -> list[BaseCallback]:
        return [
            SaveModelCallback(),
            BestEpisodeCallback(self.total_timesteps),
            UploadEpisodeCallback(self.config.upload),
            StopTrainingAtTimestepsCallback(self.total_timesteps),
        ]

    def _configure_runtime(self) -> None:
        paths = self.config.paths
        game = self.config.training.game
        wrapper = get_last_environment_wrapper()

        # noinspection PyTypeChecker
        configure_runtime(
            RetroSpeedlabRuntime(
                paths=paths,
                wrappers={game: wrapper} if wrapper else {},
                ignored_states={game: set()},
                default_states={game: self._savestate or ""},
                obs_size=(96, 96),
                action_repeat=1,
                get_game=lambda: game,
                get_savestate=lambda: self._savestate or "",
                set_savestate=self._set_savestate,
                get_state_value=self._get_state_value,
                set_state_value=self._set_state_value,
                get_model_path=lambda selected_game: get_model_path(str(paths.models_dir), selected_game),
                get_model_metadata=get_model_metadata,
            )
        )

    def _get_state_value(self, name: str) -> str:
        state_path = self._state_path(name)
        if state_path.exists():
            with state_path.open(encoding="utf-8") as file:
                return file.read()
        return str(self._state.get(name, "False"))

    def _set_state_value(self, name: str, value: Any) -> None:
        self._state[name] = value
        state_path = self._state_path(name)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as file:
            file.write(str(value))

    def _set_savestate(self, savestate: str) -> None:
        self._savestate = savestate

    def _state_path(self, name: str) -> Path:
        return Path(
            self.config.paths.models_dir,
            self.config.training.game,
            self._savestate or "",
            f"{name}.txt",
        )
