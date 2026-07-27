import math
from pathlib import Path


class ScoreStagnation:
    episodes_allowed: int = 16

    def __init__(self, root: Path) -> None:
        self.root = root

    def record(self, state: str, score: float) -> tuple[int, bool]:
        if not math.isfinite(score):
            raise ValueError(f"Curriculum score must be finite, got {score}")
        best = self._best(state)
        if best is None or score > best:
            self._best_path(state).write_text(repr(score), encoding="utf-8")
            self._episodes_path(state).unlink(missing_ok=True)
            return 0, False
        episodes = self._episodes(state) + 1
        if episodes < self.episodes_allowed:
            self._episodes_path(state).write_text(str(episodes), encoding="utf-8")
            return episodes, False
        self.clear(state)
        self._savestate_path(state).unlink(missing_ok=True)
        return episodes, True

    def clear(self, state: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._best_path(state).unlink(missing_ok=True)
        self._episodes_path(state).unlink(missing_ok=True)

    def _best(self, state: str) -> float | None:
        try:
            score = float(self._best_path(state).read_text(encoding="utf-8"))
            return score if math.isfinite(score) else None
        except (FileNotFoundError, ValueError):
            return None

    def _episodes(self, state: str) -> int:
        try:
            return max(0, int(self._episodes_path(state).read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return 0

    def _best_path(self, state: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{state}.best_score"

    def _episodes_path(self, state: str) -> Path:
        return self.root / f"{state}.stagnant_episodes"

    def _savestate_path(self, state: str) -> Path:
        return self.root / f"{state}.state"
