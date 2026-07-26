import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from datenwissenschaften.curriculum.stagnation import ScoreStagnation


class SavestateCurriculum:
    wins_required: int = 16

    def __init__(self, root: Path, states: tuple[str, ...]) -> None:
        if not states:
            raise ValueError("Curriculum requires at least one state")
        self.root: Path = root
        self.states: tuple[str, ...] = states
        self.episode_state: str = states[0]
        self.recorded: bool = False
        self.stagnation: ScoreStagnation = ScoreStagnation(root)

    def start(self) -> tuple[str, bytes | None]:
        self.recorded = False
        target_index = next(
            (index for index, state in enumerate(self.states) if self.wins(state) < self.wins_required),
            0,
        )
        for state in reversed(self.states[1 : target_index + 1]):
            path = self._savestate_path(state)
            if path.is_file():
                self.episode_state = state
                return state, path.read_bytes()
        self.episode_state = self.states[0]
        return self.episode_state, None

    def transition(self, previous: str, current: str, savestate: bytes) -> tuple[int, bool]:
        if self.recorded or previous != self.episode_state:
            return self.wins(previous), False
        self.recorded = True
        if self.wins(previous) >= self.wins_required:
            self._write(self._savestate_path(current), savestate)
            return self.wins(previous), False
        return self._record(previous, current, savestate)

    def victory(self, state: str) -> tuple[int, bool]:
        if self.recorded or state != self.episode_state:
            return self.wins(state), False
        self.recorded = True
        return self._record(state, None, None)

    def record_attempt(self, score: float) -> tuple[int, bool]:
        if self.recorded or self.episode_state == self.states[0]:
            return 0, False
        with self._lock(self.episode_state):
            return self.stagnation.record(self.episode_state, score)

    def wins(self, state: str) -> int:
        self._require_state(state)
        try:
            return min(self.wins_required, max(0, int(self._wins_path(state).read_text(encoding="utf-8"))))
        except (FileNotFoundError, ValueError):
            return 0

    def _record(self, state: str, next_state: str | None, savestate: bytes | None) -> tuple[int, bool]:
        with self._lock(state):
            wins = self.wins(state)
            if wins >= self.wins_required:
                return wins, False
            wins += 1
            mastered = wins == self.wins_required
            if mastered and next_state is not None and savestate is not None:
                self._write(self._savestate_path(next_state), savestate)
            self._write(self._wins_path(state), str(wins).encode())
            if mastered:
                self.stagnation.clear(state)
            return wins, mastered

    def _require_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown curriculum state: {state}")

    def _savestate_path(self, state: str) -> Path:
        return self.root / f"{state}.state"

    def _wins_path(self, state: str) -> Path:
        return self.root / f"{state}.wins"

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
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
