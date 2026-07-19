from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ReverseCurriculum:
    """Persistent ordered curriculum for a state sequence.

    States are mastered from first to last. A missing or rejected checkpoint
    never removes mastery, but may temporarily start from the nearest earlier
    mastered checkpoint so the environment can rebuild the bad savestate.
    """

    WIN_TARGET = 8
    BAD_CHECKPOINT_EVIDENCE_TARGET = 32

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
        for state_name in self.state_names:
            if not self.is_mastered(state_name):
                return state_name
        return None

    def has_checkpoint(self, state_name: str) -> bool:
        self._require_state(state_name)
        return self._checkpoint_path(state_name).is_file()

    def episode_start_state(self) -> str | None:
        """Return the checkpoint state to restore for the current target.

        Normally this is the unmastered target itself. If that checkpoint was
        rejected, fall back through mastered states until a usable checkpoint
        is found. ``None`` means the configured initial savestate is required.
        """
        target_state = self.active_state()
        if target_state is None:
            return None
        target_index = self.state_names.index(target_state)
        for state_name in reversed(self.state_names[1 : target_index + 1]):
            if self.has_checkpoint(state_name):
                return state_name
        return None

    def is_complete(self) -> bool:
        return all(self.is_mastered(state_name) for state_name in self.state_names)

    def checkpoint(self, state_name: str) -> bytes:
        self._require_state(state_name)
        return self._checkpoint_path(state_name).read_bytes()

    def record_success(self, state_name: str, episode_steps: int) -> bool:
        self._require_state(state_name)
        with self._lock(state_name):
            if self.is_mastered(state_name):
                return False
            self._record_longest_attempt(state_name, episode_steps)
            target = self.win_target(state_name)
            wins = min(target, self.wins(state_name) + 1)
            self._atomic_write(self._success_path(state_name), str(wins).encode("utf-8"))
            self._clear_score_evidence(state_name)
            mastered = wins >= target
            return mastered

    def record_failure(self, state_name: str, episode_steps: int, score: float) -> bool:
        self._require_state(state_name)
        with self._lock(state_name):
            if self.is_mastered(state_name):
                return False
            self._record_longest_attempt(state_name, episode_steps)
            checkpoint = self._checkpoint_path(state_name)
            if state_name == self.state_names[0] or not checkpoint.is_file():
                return False

            score = float(score)
            if not math.isfinite(score):
                raise ValueError(f"Curriculum score must be finite, got {score}")
            best_score = self.best_score(state_name)
            last_score = self.last_score(state_name)
            self._atomic_write(self._last_score_path(state_name), repr(score).encode("utf-8"))

            if best_score is None or score > best_score:
                self._atomic_write(self._best_score_path(state_name), repr(score).encode("utf-8"))
                self._evidence_path(state_name).unlink(missing_ok=True)
                return False

            evidence_added = 2 if last_score is not None and score < last_score else 1
            evidence = self.stagnation_evidence(state_name) + evidence_added
            if evidence < self.bad_checkpoint_evidence_target(state_name):
                self._atomic_write(self._evidence_path(state_name), str(evidence).encode("utf-8"))
                return False

            checkpoint.unlink(missing_ok=True)
            self._clear_score_evidence(state_name)
            return True

    def wins(self, state_name: str) -> int:
        self._require_state(state_name)
        try:
            persisted_wins = int(self._success_path(state_name).read_text(encoding="utf-8").strip())
            return min(self.WIN_TARGET, max(0, persisted_wins))
        except (FileNotFoundError, ValueError):
            return 0

    def is_mastered(self, state_name: str) -> bool:
        return self.wins(state_name) >= self.win_target(state_name)

    def typical_steps(self, state_name: str) -> int:
        self._require_state(state_name)
        return max(1, self._read_int(self._attempt_steps_path(state_name)))

    def win_target(self, state_name: str) -> int:
        self._require_state(state_name)
        return self.WIN_TARGET

    def bad_checkpoint_evidence_target(self, state_name: str) -> int:
        self._require_state(state_name)
        return self.BAD_CHECKPOINT_EVIDENCE_TARGET

    def stagnation_evidence(self, state_name: str) -> int:
        self._require_state(state_name)
        try:
            return max(0, int(self._evidence_path(state_name).read_text(encoding="utf-8").strip()))
        except (FileNotFoundError, ValueError):
            return 0

    def best_score(self, state_name: str) -> float | None:
        self._require_state(state_name)
        return self._read_float(self._best_score_path(state_name))

    def last_score(self, state_name: str) -> float | None:
        self._require_state(state_name)
        return self._read_float(self._last_score_path(state_name))

    def progress(self) -> dict[str, dict[str, int | float | bool | None]]:
        active_state = self.active_state()
        return {
            state_name: {
                "wins": self.wins(state_name),
                "win_target": self.win_target(state_name),
                "bad_checkpoint_evidence": self.stagnation_evidence(state_name),
                "bad_checkpoint_evidence_target": self.bad_checkpoint_evidence_target(state_name),
                "best_checkpoint_score": self.best_score(state_name),
                "last_checkpoint_score": self.last_score(state_name),
                "typical_episode_steps": self.typical_steps(state_name),
                "mastered": self.is_mastered(state_name),
                "has_checkpoint": self.has_checkpoint(state_name),
                "active": state_name == active_state,
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

    def _evidence_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.score_stagnation"

    def _best_score_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.best_score"

    def _last_score_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.last_score"

    def _attempt_steps_path(self, state_name: str) -> Path:
        return self.root / f"{state_name}.attempt_steps"

    def _record_longest_attempt(self, state_name: str, episode_steps: int) -> None:
        longest = max(1, episode_steps, self._read_int(self._attempt_steps_path(state_name)))
        self._atomic_write(self._attempt_steps_path(state_name), str(longest).encode("utf-8"))

    @staticmethod
    def _read_int(path: Path) -> int:
        try:
            return max(0, int(path.read_text(encoding="utf-8").strip()))
        except (FileNotFoundError, ValueError):
            return 0

    @staticmethod
    def _read_float(path: Path) -> float | None:
        try:
            value = float(path.read_text(encoding="utf-8").strip())
            return value if math.isfinite(value) else None
        except (FileNotFoundError, ValueError):
            return None

    def _clear_score_evidence(self, state_name: str) -> None:
        self._evidence_path(state_name).unlink(missing_ok=True)
        self._best_score_path(state_name).unlink(missing_ok=True)
        self._last_score_path(state_name).unlink(missing_ok=True)

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
