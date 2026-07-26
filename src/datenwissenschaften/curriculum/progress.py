import os
from pathlib import Path


class SavestateCurriculum:
    def __init__(self, root: Path, states: tuple[str, ...]) -> None:
        if not states:
            raise ValueError("Curriculum requires at least one state")
        self.root: Path = root
        self.states: tuple[str, ...] = states
        self.episode_state: str = states[0]
        self.recorded: bool = False

    def start(self) -> tuple[str, bytes | None]:
        self.recorded = False
        self.episode_state = self.states[0]
        for state in self.states:
            if self._completed_path(state).is_file():
                continue
            self.episode_state = state
            path = self._savestate_path(state)
            return state, path.read_bytes() if path.is_file() else None
        return self.states[0], None

    def transition(self, previous: str, current: str, savestate: bytes) -> bool:
        if self.recorded or previous != self.episode_state:
            return False
        self._require_state(current)
        self.recorded = True
        self._write(self._savestate_path(current), savestate)
        self._write(self._completed_path(previous), b"")
        return True

    def victory(self, state: str) -> bool:
        if self.recorded or state != self.episode_state:
            return False
        self.recorded = True
        self._write(self._completed_path(state), b"")
        return True

    def _require_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown curriculum state: {state}")

    def _savestate_path(self, state: str) -> Path:
        return self.root / f"{state}.state"

    def _completed_path(self, state: str) -> Path:
        return self.root / f"{state}.complete"

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
