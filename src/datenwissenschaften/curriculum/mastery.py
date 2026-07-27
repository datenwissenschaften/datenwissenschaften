from pathlib import Path


class StageMastery:
    def __init__(self, root: Path, required_successes: int) -> None:
        if required_successes < 1:
            raise ValueError("required_successes must be positive")
        self.root = root
        self.required_successes = required_successes

    def record(self, state: str) -> tuple[int, bool]:
        successes = self._successes(state) + 1
        if successes >= self.required_successes:
            self.clear(state)
            return successes, True
        self._path(state).write_text(str(successes), encoding="utf-8")
        return successes, False

    def clear(self, state: str) -> None:
        self._path(state).unlink(missing_ok=True)

    def _successes(self, state: str) -> int:
        try:
            return max(0, int(self._path(state).read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError):
            return 0

    def _path(self, state: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{state}.successes"
