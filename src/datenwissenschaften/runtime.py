from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datenwissenschaften.retro.paths import RetroArenaPaths


@dataclass(frozen=True)
class RetroArenaRuntime:
    paths: RetroArenaPaths
    wrappers: Mapping[str, type]
    ignored_states: Mapping[str, set[str]]
    default_states: Mapping[str, str]
    obs_size: tuple[int, int]
    action_repeat: int
    get_game: Callable[[], str]
    get_savestate: Callable[[], str]
    set_savestate: Callable[[str], None]
    get_state_value: Callable[[str], str]
    set_state_value: Callable[[str, Any], None]
    get_model_path: Callable[[str], str]

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
    def working_dir(self) -> Path:
        return self.paths.working_dir

    @property
    def record_dir(self) -> Path:
        return self.paths.record_dir


_runtime: RetroArenaRuntime | None = None


def configure_runtime(runtime: RetroArenaRuntime) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> RetroArenaRuntime:
    if _runtime is None:
        raise RuntimeError("datenwissenschaften runtime is not configured.")
    return _runtime
