from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks import (
    BestEpisodeCallback,
    SaveModelCallback,
    StopTrainingAtTimestepsCallback,
)
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.model import get_model_metadata, get_model_path
from datenwissenschaften.retro.environment import get_last_environment_wrapper
from datenwissenschaften.runtime import RetroSpeedlabRuntime, configure_runtime
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, RetroSpeedlabConfig, load_config


class Trainer:
    # noinspection PyTypeChecker
    def __init__(
        self,
        *,
        additional_callbacks: Sequence[BaseCallback] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config: RetroSpeedlabConfig = load_config()
        self.total_timesteps = self.config.training.total_timesteps
        self.callbacks = self._default_callbacks() + (additional_callbacks or [])
        self._state: dict[str, Any] = {}
        self._savestate = self.config.training.savestate

    def train(self, model) -> None:
        self._configure_runtime()
        if model.num_timesteps >= self.total_timesteps:
            return

        model.learn(
            total_timesteps=self.total_timesteps - model.num_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
        )

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
