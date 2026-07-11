from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.callbacks import (
    BestEpisodeCallback,
    EpisodeTelemetryCallback,
    ModelMetadataCallback,
    SaveModelCallback,
)
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.model import get_model_metadata, get_model_path
from datenwissenschaften.persistence import RedisStore
from datenwissenschaften.retro.environment import get_last_environment_wrapper
from datenwissenschaften.runtime import RetroSpeedlabRuntime, configure_runtime
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, RetroSpeedlabConfig, load_config
from datenwissenschaften.ui import configure_history, publish_metadata, start_ui
from datenwissenschaften.ui.control import configure_training_control


class Trainer:
    training_chunk_steps = 10_000_000

    # noinspection PyTypeChecker
    def __init__(
        self,
        *,
        additional_callbacks: Sequence[BaseCallback] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        state_name: str,
    ) -> None:
        self.config: RetroSpeedlabConfig = load_config(config_path)
        setup_logging(self.config.log_level)
        self.state_name = state_name
        self._additional_callbacks = list(additional_callbacks or [])
        self.callbacks = self._default_callbacks() + self._additional_callbacks
        self._savestate = self.config.training.active_savestate
        self._store = RedisStore(self.config.ui.redis_url)

    def train(self, model) -> None:
        configure_accelerator()
        self._configure_runtime()
        self._start_ui(model)
        while True:
            model.learn(
                total_timesteps=self.training_chunk_steps,
                callback=self.callbacks,
                reset_num_timesteps=False,
            )

    def _start_ui(self, model) -> None:
        if not self.config.ui.enabled:
            return
        history_scope = self.config.training.game_identity
        configure_history(
            history_scope,
            redis_url=self.config.ui.redis_url,
            key_prefix=self.config.ui.history_key_prefix,
        )
        model_dir = self.config.paths.models_dir / self.config.training.game_identity / self.state_name
        configure_training_control(
            game=self.config.training.game,
            model_dir=model_dir,
            restart_supported=bool(getattr(model, "supports_ui_restart", False)),
            on_reset=lambda: self._reset_for_restart(model),
        )
        if start_ui(self.config.ui) is None:
            return
        env = model.get_env() if callable(getattr(model, "get_env", None)) else getattr(model, "env", None)
        publish_metadata(
            "run",
            {
                "game": self.config.training.game,
                "game_identity": self.config.training.game_identity,
                "training_state": self.state_name,
                "savestate": self._savestate,
                "savestates": list(self.config.training.savestates),
                "configured_envs": self.config.training.num_envs,
            },
        )
        model_metadata = get_model_metadata(model)
        publish_metadata("model", model_metadata, replace=True)
        publish_metadata("environment", self._environment_metadata(env))

    def _reset_for_restart(self, model) -> None:
        self._savestate = self.config.training.active_savestate
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
        callbacks = [
            ModelMetadataCallback(),
            SaveModelCallback(),
            EpisodeTelemetryCallback(),
            BestEpisodeCallback(),
            UploadEpisodeCallback(self.config.upload),
        ]
        return [callback for callback in callbacks if callback is not None]

    def _configure_runtime(self) -> None:
        paths = self.config.paths
        game = self.config.training.game
        game_identity = self.config.training.game_identity
        wrapper = get_last_environment_wrapper()

        # noinspection PyTypeChecker
        configure_runtime(
            RetroSpeedlabRuntime(
                paths=paths,
                wrappers={game: wrapper} if wrapper else {},
                ignored_states={game: set()},
                default_states={game: self._savestate or ""},
                obs_size=(96, 96),
                get_game=lambda: game,
                get_savestate=lambda: self._savestate or "",
                set_savestate=self._set_savestate,
                get_state_value=self._get_state_value,
                set_state_value=self._set_state_value,
                get_model_path=lambda _selected_game: get_model_path(
                    str(paths.models_dir),
                    game_identity,
                    self.state_name,
                ),
                get_model_metadata=get_model_metadata,
            )
        )

    def _get_state_value(self, name: str) -> Any:
        return self._store.get(*self._state_key(name), default=False)

    def _set_state_value(self, name: str, value: Any) -> None:
        self._store.set(*self._state_key(name), value=value)

    def _set_savestate(self, savestate: str) -> None:
        self._savestate = savestate

    def _state_key(self, name: str) -> tuple[str, ...]:
        # Runtime state belongs to the emulator/game state, not to the training objective.
        return ("state", self.config.training.game_identity, self._savestate or "default", name)
