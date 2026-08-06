from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ReverseCurriculum:
    WIN_TARGET = 64
    BAD_CHECKPOINT_EVIDENCE_TARGET = 128

    def __init__(self, root: Path, state_names: Sequence[str]) -> None:
        if not state_names:
            raise ValueError("Curriculum requires at least one state")
        self.root = root
        self.state_names = tuple(state_names)

    def active_state(self) -> str | None:
        return next((name for name in self.state_names if not self.is_mastered(name)), None)

    def episode_start_state(self) -> str | None:
        target = self.active_state()
        if target is None:
            return None
        target_index = self.state_names.index(target)
        return next(
            (name for name in reversed(self.state_names[1 : target_index + 1]) if self.has_checkpoint(name)),
            None,
        )

    def has_checkpoint(self, state_name: str) -> bool:
        self._require_state(state_name)
        return self._checkpoint_path(state_name).is_file()

    def checkpoint(self, state_name: str) -> bytes:
        self._require_state(state_name)
        return self._checkpoint_path(state_name).read_bytes()

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

    def record_success(self, state_name: str, episode_steps: int) -> bool:
        self._require_state(state_name)
        with self._lock(state_name):
            if self.is_mastered(state_name):
                return False
            self._write_int(self._steps_path(state_name), max(1, episode_steps, self.typical_steps(state_name)))
            wins = min(self.WIN_TARGET, self.wins(state_name) + 1)
            self._write_int(self._success_path(state_name), wins)
            self._clear_failure_evidence(state_name)
            return wins >= self.WIN_TARGET

    def record_failure(self, state_name: str, episode_steps: int, score: float) -> bool:
        self._require_state(state_name)
        if not math.isfinite(score):
            raise ValueError(f"Curriculum score must be finite, got {score}")
        self._write_int(self._steps_path(state_name), max(1, episode_steps, self.typical_steps(state_name)))
        checkpoint = self._checkpoint_path(state_name)
        if state_name == self.state_names[0] or not checkpoint.is_file():
            return False
        best = self._read_float(self._best_score_path(state_name))
        last = self._read_float(self._last_score_path(state_name))
        self._write_float(self._last_score_path(state_name), score)
        if best is None or score > best:
            self._write_float(self._best_score_path(state_name), score)
            self._evidence_path(state_name).unlink(missing_ok=True)
            return False
        evidence = self._read_int(self._evidence_path(state_name)) + (2 if last is not None and score < last else 1)
        if evidence < self.BAD_CHECKPOINT_EVIDENCE_TARGET:
            self._write_int(self._evidence_path(state_name), evidence)
            return False
        checkpoint.unlink(missing_ok=True)
        self._clear_failure_evidence(state_name)
        return True

    def wins(self, state_name: str) -> int:
        self._require_state(state_name)
        return min(self.WIN_TARGET, self._read_int(self._success_path(state_name)))

    def is_mastered(self, state_name: str) -> bool:
        return self.wins(state_name) >= self.WIN_TARGET

    def is_complete(self) -> bool:
        return all(self.is_mastered(name) for name in self.state_names)

    def typical_steps(self, state_name: str) -> int:
        self._require_state(state_name)
        return max(1, self._read_int(self._steps_path(state_name)))

    def progress(self) -> dict[str, dict[str, int | float | bool | None]]:
        active = self.active_state()
        return {
            name: {
                "wins": self.wins(name),
                "win_target": self.WIN_TARGET,
                "mastered": self.is_mastered(name),
                "has_checkpoint": self.has_checkpoint(name),
                "active": name == active,
                "typical_episode_steps": self.typical_steps(name),
            }
            for name in self.state_names
        }

    def _require_state(self, state_name: str) -> None:
        if state_name not in self.state_names:
            raise ValueError(f"Unknown curriculum state: {state_name}")

    def _checkpoint_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.state"

    def _success_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.successes"

    def _steps_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.attempt_steps"

    def _evidence_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.score_stagnation"

    def _best_score_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.best_score"

    def _last_score_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.last_score"

    def _clear_failure_evidence(self, state_name: str) -> None:
        for path in (
            self._evidence_path(state_name),
            self._best_score_path(state_name),
            self._last_score_path(state_name),
        ):
            path.unlink(missing_ok=True)

    def _write_int(self, path: Path, value: int) -> None:
        self._atomic_write(path, str(value).encode())

    def _write_float(self, path: Path, value: float) -> None:
        self._atomic_write(path, repr(value).encode())

    @staticmethod
    def _read_int(path: Path) -> int:
        try:
            return max(0, int(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return 0

    @staticmethod
    def _read_float(path: Path) -> float | None:
        try:
            value = float(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None
        return value if math.isfinite(value) else None

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
