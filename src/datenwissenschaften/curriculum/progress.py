import fcntl
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from datenwissenschaften.curriculum.stagnation import ScoreStagnation


class SavestateCurriculum:
    def __init__(self, root: Path, states: tuple[str, ...]) -> None:
        if not states:
            raise ValueError("Curriculum requires at least one state")
        self.root = root
        self.states = states
        self.episode_state = states[0]
        self.recorded = False
        self.stagnation: ScoreStagnation = ScoreStagnation(root)

    def start(self) -> tuple[str, bytes | None]:
        self.recorded = False
        target_index = next(
            (index for index, state in enumerate(self.states) if not self._completed_path(state).is_file()),
            0,
        )
        for state in reversed(self.states[1 : target_index + 1]):
            savestate = self._savestate_path(state)
            if savestate.is_file():
                self.episode_state = state
                return state, savestate.read_bytes()
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

    def record_attempt(self, score: float) -> tuple[int, bool]:
        if self.recorded or self.episode_state == self.states[0]:
            return 0, False
        with self._lock(self.episode_state):
            return self.stagnation.record(self.episode_state, score)

    def _require_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown curriculum state: {state}")

    def _savestate_path(self, state: str) -> Path:
        return self.root / f"{state}.state"

    def _completed_path(self, state: str) -> Path:
        return self.root / f"{state}.complete"

    @contextmanager
    def _lock(self, state: str) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f".{state}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / path.name
            temporary.write_bytes(content)
            temporary.replace(path)
