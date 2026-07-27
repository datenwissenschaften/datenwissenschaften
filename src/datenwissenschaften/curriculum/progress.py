import tempfile
from pathlib import Path


class SavestateCurriculum:
    def __init__(self, root: Path, states: tuple[str, ...]) -> None:
        if not states:
            raise ValueError("Curriculum requires at least one state")
        self.root = root
        self.states = states
        self.episode_state = states[0]
        self.recorded = False

    def start(self) -> tuple[str, bytes | None]:
        self.recorded = False
        for state in self.states:
            if self._completed_path(state).is_file():
                continue
            self.episode_state = state
            savestate = self._savestate_path(state)
            return state, savestate.read_bytes() if savestate.is_file() else None
        self.episode_state = self.states[0]
        return self.episode_state, None

    def transition(self, previous: str, current: str, savestate: bytes) -> bool:
        self._require_state(current)
        if self.recorded or previous != self.episode_state:
            return False
        self._write(self._savestate_path(current), savestate)
        self._write(self._completed_path(previous), b"")
        self.recorded = True
        return True

    def victory(self, state: str) -> bool:
        self._require_state(state)
        if self.recorded or state != self.episode_state:
            return False
        self._write(self._completed_path(state), b"")
        self.recorded = True
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
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / path.name
            temporary.write_bytes(content)
            temporary.replace(path)
