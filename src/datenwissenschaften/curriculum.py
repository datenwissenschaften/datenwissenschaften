from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ReverseCurriculum:
    """Persistent deepest-checkpoint-first curriculum for a state sequence."""

    def __init__(self, root: Path, state_names: Sequence[str]) -> None:
        self.root = root
        self.state_names = tuple(state_names)

    def save_checkpoint(self, state_name: str, emulator_state: bytes) -> bool:
        self._require_state(state_name)
        if self.is_mastered(state_name):
            return False
        path = self._checkpoint_path(state_name)
        with self._lock(state_name):
            if path.is_file():
                return False
            self._atomic_write(path, emulator_state)
        return True

    def active_state(self) -> str | None:
        for state_name in reversed(self.state_names[1:]):
            if not self.is_mastered(state_name) and self._checkpoint_path(state_name).is_file():
                return state_name
        return None

    def checkpoint(self, state_name: str) -> bytes:
        self._require_state(state_name)
        return self._checkpoint_path(state_name).read_bytes()

    def record_success(self, state_name: str, episode_steps: int) -> bool:
        self._require_state(state_name)
        with self._lock(state_name):
            self._record_longest_attempt(state_name, episode_steps)
            longest_success = max(episode_steps, self._read_int(self._success_steps_path(state_name)))
            self._atomic_write(self._success_steps_path(state_name), str(longest_success).encode("utf-8"))
            threshold = self.success_threshold(state_name)
            successes = min(threshold, self.successes(state_name) + 1)
            self._atomic_write(self._success_path(state_name), str(successes).encode("utf-8"))
            self._failure_path(state_name).unlink(missing_ok=True)
            mastered = successes >= threshold
            if mastered:
                self._checkpoint_path(state_name).unlink(missing_ok=True)
            return mastered

    def record_failure(self, state_name: str, episode_steps: int) -> bool:
        self._require_state(state_name)
        with self._lock(state_name):
            if self.is_mastered(state_name):
                return False
            self._success_path(state_name).unlink(missing_ok=True)
            self._record_longest_attempt(state_name, episode_steps)
            checkpoint = self._checkpoint_path(state_name)
            if state_name == self.state_names[0] or not checkpoint.is_file():
                return False
            typical_steps = self.typical_steps(state_name)
            severity = max(1, math.ceil(typical_steps / max(1, episode_steps)))
            failures = self.failures(state_name) + severity
            if failures < self.failure_threshold(state_name):
                self._atomic_write(self._failure_path(state_name), str(failures).encode("utf-8"))
                return False
            checkpoint.unlink(missing_ok=True)
            self._failure_path(state_name).unlink(missing_ok=True)
            return True

    def successes(self, state_name: str) -> int:
        self._require_state(state_name)
        try:
            return max(0, int(self._success_path(state_name).read_text(encoding="utf-8").strip()))
        except (FileNotFoundError, ValueError):
            return 0

    def is_mastered(self, state_name: str) -> bool:
        return self.successes(state_name) >= self.success_threshold(state_name)

    def typical_steps(self, state_name: str) -> int:
        self._require_state(state_name)
        return max(1, self._read_int(self._attempt_steps_path(state_name)))

    def success_threshold(self, state_name: str) -> int:
        successful_steps = self._read_int(self._success_steps_path(state_name))
        typical_steps = successful_steps or self.typical_steps(state_name)
        return max(2, math.ceil(math.log2(typical_steps + 1)) - 1)

    def failure_threshold(self, state_name: str) -> int:
        typical_steps = self.typical_steps(state_name)
        return math.ceil(2 * math.sqrt(typical_steps) + math.log2(typical_steps + 1))

    def failures(self, state_name: str) -> int:
        self._require_state(state_name)
        try:
            return max(0, int(self._failure_path(state_name).read_text(encoding="utf-8").strip()))
        except (FileNotFoundError, ValueError):
            return 0

    def progress(self) -> dict[str, dict[str, int | bool]]:
        active_state = self.active_state()
        return {
            state_name: {
                "consecutive_successes": self.successes(state_name),
                "success_threshold": self.success_threshold(state_name),
                "bad_checkpoint_evidence": self.failures(state_name),
                "failure_threshold": self.failure_threshold(state_name),
                "typical_episode_steps": self.typical_steps(state_name),
                "mastered": self.is_mastered(state_name),
                "has_checkpoint": self._checkpoint_path(state_name).is_file(),
                "active": state_name == (active_state or self.state_names[0]),
            }
            for state_name in self.state_names
        }

    def _require_state(self, state_name: str) -> None:
        if state_name not in self.state_names:
            raise ValueError(f"Unknown curriculum state: {state_name}")

    def _checkpoint_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.state"

    def _success_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.successes"

    def _failure_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.failures"

    def _attempt_steps_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.attempt_steps"

    def _success_steps_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.success_steps"

    def _record_longest_attempt(self, state_name: str, episode_steps: int) -> None:
        longest = max(1, episode_steps, self._read_int(self._attempt_steps_path(state_name)))
        self._atomic_write(self._attempt_steps_path(state_name), str(longest).encode("utf-8"))

    @staticmethod
    def _read_int(path: Path) -> int:
        try:
            return max(0, int(path.read_text(encoding="utf-8").strip()))
        except (FileNotFoundError, ValueError):
            return 0

    @contextmanager
    def _lock(self, state_name: str) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f".{state_name}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
