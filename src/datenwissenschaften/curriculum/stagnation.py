import math
from pathlib import Path


class ScoreStagnation:
    attempts_allowed: int = 128

    def __init__(self, root: Path) -> None:
        self.root: Path = root

    def record(self, state: str, score: float) -> tuple[int, bool]:
        if not math.isfinite(score):
            raise ValueError(f"Curriculum score must be finite, got {score}")
        best = self._best(state)
        if best is None or score > best:
            self._best_path(state).write_text(repr(score), encoding="utf-8")
            self._attempts_path(state).unlink(missing_ok=True)
            return 0, False
        attempts = self._attempts(state) + 1
        if attempts < self.attempts_allowed:
            self._attempts_path(state).write_text(str(attempts), encoding="utf-8")
            return attempts, False
        self.clear(state)
        self._savestate_path(state).unlink(missing_ok=True)
        return attempts, True

    def clear(self, state: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._best_path(state).unlink(missing_ok=True)
        self._attempts_path(state).unlink(missing_ok=True)

    def _best(self, state: str) -> float | None:
        try:
            value = float(self._best_path(state).read_text(encoding="utf-8"))
            return value if math.isfinite(value) else None
        except (FileNotFoundError, ValueError):
            return None

    def _attempts(self, state: str) -> int:
        try:
            return max(0, int(self._attempts_path(state).read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return 0

    def _best_path(self, state: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{state}.best_score"

    def _attempts_path(self, state: str) -> Path:
        return self.root / f"{state}.stagnant_attempts"

    def _savestate_path(self, state: str) -> Path:
        return self.root / f"{state}.state"
