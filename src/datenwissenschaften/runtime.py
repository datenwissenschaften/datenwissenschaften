from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datenwissenschaften.retro.paths import RetroSpeedlabPaths


@dataclass(frozen=True)
class RetroSpeedlabRuntime:
    paths: RetroSpeedlabPaths
    wrappers: Mapping[str, type]
    ignored_states: Mapping[str, set[str]]
    default_states: Mapping[str, str]
    obs_size: tuple[int, int]
    get_game: Callable[[], str]
    get_savestate: Callable[[], str]
    set_savestate: Callable[[str], None]
    get_state_value: Callable[[str], str]
    set_state_value: Callable[[str, Any], None]
    get_model_path: Callable[[str], str]
    get_model_metadata: Callable[[Any], dict[str, Any]]

    @property
    def game(self) -> str:
        return self.get_game()

    @property
    def savestate(self) -> str:
        return self.get_savestate()

    @property
    def models_dir(self) -> Path:
        return self.paths.models_dir

    @property
    def record_dir(self) -> Path:
        return self.paths.record_dir

    @property
    def cache_dir(self) -> Path:
        return self.paths.cache_dir


_runtime: RetroSpeedlabRuntime | None = None


def configure_runtime(runtime: RetroSpeedlabRuntime) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> RetroSpeedlabRuntime:
    if _runtime is None:
        raise RuntimeError("datenwissenschaften runtime is not configured.")
    return _runtime
