from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym


@dataclass(frozen=True)
class GameDefinition:
    game_id: str
    wrapper: type
    default_state: str | None = None
    ignore_states: set[str] | None = None


class GameDefinitionLoader:
    def __init__(self, game_dir: Path) -> None:
        self.game_dir = game_dir

    def load(self) -> GameDefinition:
        if not self.game_dir.is_dir():
            raise ValueError(f"GAME_DIR does not exist or is not a directory: {self.game_dir}")

        package_name = self.game_dir.name
        parent = str(self.game_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        states_module = importlib.import_module(f"{package_name}.states")
        game_id = getattr(states_module, "GAME_ID", None)
        if not game_id:
            raise ValueError(f"{self.game_dir}/states.py must define GAME_ID.")

        package = importlib.import_module(package_name)
        wrapper = self._find_wrapper(package)

        # noinspection PyTypeChecker
        return GameDefinition(
            game_id=game_id,
            wrapper=wrapper,
            default_state=getattr(states_module, "DEFAULT_STATE", None),
            ignore_states=set(getattr(states_module, "IGNORE_STATES", []) or []),
        )

    @staticmethod
    def _find_wrapper(package) -> type:
        for _, obj in inspect.getmembers(package, inspect.isclass):
            if issubclass(obj, gym.Wrapper) and obj is not gym.Wrapper:
                return obj
        raise ValueError(f"{package.__name__} must export a gymnasium.Wrapper subclass.")


class GameRegistry:
    def __init__(self, definition: GameDefinition) -> None:
        self.definition = definition

    @classmethod
    def from_game_dir(cls, game_dir: Path) -> "GameRegistry":
        return cls(GameDefinitionLoader(game_dir).load())

    @property
    def wrappers(self) -> dict[str, type]:
        return {self.definition.game_id: self.definition.wrapper}

    @property
    def default_states(self) -> dict[str, str]:
        if not self.definition.default_state:
            return {}
        return {self.definition.game_id: self.definition.default_state}

    @property
    def ignored_states(self) -> dict[str, set[str]]:
        return {self.definition.game_id: self.definition.ignore_states or set()}
