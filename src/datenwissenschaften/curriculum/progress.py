from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.curriculum.mastery import StageMastery
from datenwissenschaften.curriculum.storage import CurriculumStorage


@dataclass(frozen=True, slots=True)
class SavestateDiagnostics:
    state: str
    attempts: int
    recent_median: float
    best_median: float
    trend: float
    progress_rate: float
    stagnant_windows: int
    deleted: bool

    def __iter__(self):
        yield self.attempts
        yield self.deleted


@dataclass(slots=True)
class SavestatePerformance:
    scores: deque[float]
    progressed: deque[bool]
    best_median: float
    stagnant_windows: int

    @classmethod
    def create(cls, history_size: int) -> SavestatePerformance:
        return cls(
            scores=deque(maxlen=history_size),
            progressed=deque(maxlen=history_size),
            best_median=-np.inf,
            stagnant_windows=0,
        )


class AdaptiveBadSavestateDetector:
    def __init__(
        self,
        root: Path,
        minimum_attempts: int,
        recent_fraction: float,
        patience_windows: int,
        history_size: int,
    ) -> None:
        if minimum_attempts < 4:
            raise ValueError("minimum_attempts must be at least four")
        if not 0.0 < recent_fraction <= 0.5:
            raise ValueError("recent_fraction must be between zero and 0.5")
        if patience_windows < 1:
            raise ValueError("patience_windows must be positive")
        if history_size < minimum_attempts:
            raise ValueError("history_size must be at least minimum_attempts")

        self.root = root / "stagnation"
        self.minimum_attempts = minimum_attempts
        self.recent_fraction = recent_fraction
        self.patience_windows = patience_windows
        self.history_size = history_size
        self.root.mkdir(parents=True, exist_ok=True)

        self.performance: dict[str, SavestatePerformance] = {}

    @classmethod
    def from_stagnation_episodes(
        cls,
        root: Path,
        stagnation_episodes: int,
    ) -> AdaptiveBadSavestateDetector:
        if stagnation_episodes < 4:
            raise ValueError("stagnation_episodes must be at least four")

        return cls(
            root=root,
            minimum_attempts=max(8, stagnation_episodes // 4),
            recent_fraction=0.25,
            patience_windows=max(2, stagnation_episodes // 8),
            history_size=max(64, stagnation_episodes * 2),
        )

    def record(
        self,
        state: str,
        score: float,
        progressed: bool,
    ) -> SavestateDiagnostics:
        performance = self._performance(state)
        performance.scores.append(float(score))
        performance.progressed.append(bool(progressed))

        scores = np.asarray(performance.scores, dtype=np.float64)
        progression = np.asarray(performance.progressed, dtype=np.bool_)
        attempts = len(scores)

        recent_median = float(np.median(scores))
        progress_rate = float(np.mean(progression))

        if attempts < self.minimum_attempts:
            diagnostics = SavestateDiagnostics(
                state=state,
                attempts=attempts,
                recent_median=recent_median,
                best_median=_finite_or_default(
                    performance.best_median,
                    recent_median,
                ),
                trend=0.0,
                progress_rate=progress_rate,
                stagnant_windows=performance.stagnant_windows,
                deleted=False,
            )
            self._save(state, performance)
            return diagnostics

        window_size = max(
            4,
            min(
                attempts,
                round(attempts * self.recent_fraction),
            ),
        )

        recent_scores = scores[-window_size:]
        recent_progression = progression[-window_size:]

        recent_median = float(np.median(recent_scores))
        progress_rate = float(np.mean(recent_progression))
        trend = _theil_sen_trend(recent_scores)
        scale = _robust_scale(scores)

        if not np.isfinite(performance.best_median):
            performance.best_median = recent_median
            performance.stagnant_windows = 0
        elif recent_median > performance.best_median + scale:
            performance.best_median = recent_median
            performance.stagnant_windows = 0
        elif _is_stagnant(
            recent_median=recent_median,
            best_median=performance.best_median,
            trend=trend,
            progressed=recent_progression,
            scale=scale,
        ):
            performance.stagnant_windows += 1
        else:
            performance.stagnant_windows = 0

        diagnostics = SavestateDiagnostics(
            state=state,
            attempts=attempts,
            recent_median=recent_median,
            best_median=performance.best_median,
            trend=trend,
            progress_rate=progress_rate,
            stagnant_windows=performance.stagnant_windows,
            deleted=performance.stagnant_windows >= self.patience_windows,
        )

        self._save(state, performance)
        return diagnostics

    def clear(self, state: str) -> None:
        self.performance.pop(state, None)
        self._path(state).unlink(missing_ok=True)

    def _performance(self, state: str) -> SavestatePerformance:
        if state not in self.performance:
            self.performance[state] = self._load(state)
        return self.performance[state]

    def _load(self, state: str) -> SavestatePerformance:
        path = self._path(state)

        if not path.is_file():
            return SavestatePerformance.create(self.history_size)

        try:
            data = json.loads(path.read_text())
            scores = deque(
                (float(score) for score in data["scores"]),
                maxlen=self.history_size,
            )
            progressed = deque(
                (bool(value) for value in data["progressed"]),
                maxlen=self.history_size,
            )

            if len(scores) != len(progressed):
                raise ValueError("Score and progression history lengths differ")

            return SavestatePerformance(
                scores=scores,
                progressed=progressed,
                best_median=float(data["best_median"]),
                stagnant_windows=int(data["stagnant_windows"]),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return SavestatePerformance.create(self.history_size)

    def _save(
        self,
        state: str,
        performance: SavestatePerformance,
    ) -> None:
        path = self._path(state)
        temporary = path.with_suffix(".tmp")

        data = {
            "scores": list(performance.scores),
            "progressed": list(performance.progressed),
            "best_median": performance.best_median,
            "stagnant_windows": performance.stagnant_windows,
        }

        temporary.write_text(json.dumps(data, separators=(",", ":")))
        temporary.replace(path)

    def _path(self, state: str) -> Path:
        return self.root / f"{state}.json"


class SavestateCurriculum:
    def __init__(
        self,
        root: Path,
        states: tuple[str, ...],
        required_successes: int,
        stagnation_episodes: int,
        full_run_probability: float,
    ) -> None:
        if not states:
            raise ValueError("Curriculum requires at least one state")
        if not 0.0 <= full_run_probability <= 1.0:
            raise ValueError(
                "full_run_probability must be between zero and one"
            )

        self.root = root
        self.states = states
        self.full_run_probability = full_run_probability
        self.episode_state = states[0]
        self.recorded = False
        self.training = True
        self.full_run = False

        self.mastery = StageMastery(root, required_successes)
        self.stagnation = AdaptiveBadSavestateDetector.from_stagnation_episodes(
            root,
            stagnation_episodes,
        )
        self.storage = CurriculumStorage(root)

    def start(self) -> tuple[str, bytes | None]:
        self.recorded = False
        self.full_run = False

        target_index = next(
            (
                index
                for index, state in enumerate(self.states)
                if not self.storage.completed(state)
            ),
            None,
        )

        if target_index is None:
            self.training = False
            return self._practice_or_full_run()

        self.training = True

        for state in reversed(self.states[1 : target_index + 1]):
            savestate = self.storage.savestate(state)

            if savestate.is_file():
                self.episode_state = state
                return state, savestate.read_bytes()

        self.episode_state = self.states[0]
        return self.episode_state, None

    def transition(
        self,
        previous: str,
        current: str,
        savestate: bytes,
    ) -> tuple[int, bool]:
        self._require_state(previous)
        self._require_state(current)

        if (
            not self.training
            or self.recorded
            or previous != self.episode_state
        ):
            return 0, False

        with self.storage.lock(previous):
            self.storage.save(current, savestate)
            successes, completed = self._record_success(previous)

        self.recorded = True
        return successes, completed

    def victory(self, state: str) -> tuple[int, bool]:
        self._require_state(state)

        if (
            not self.training
            or self.recorded
            or state != self.episode_state
        ):
            return 0, False

        with self.storage.lock(state):
            successes, completed = self._record_success(state)

        self.recorded = True
        return successes, completed

    def record_attempt(
        self,
        score: float,
        progressed: bool,
    ) -> SavestateDiagnostics:
        if (
            not self.training
            or self.recorded
            or self.episode_state == self.states[0]
        ):
            return SavestateDiagnostics(
                state=self.episode_state,
                attempts=0,
                recent_median=float(score),
                best_median=float(score),
                trend=0.0,
                progress_rate=float(progressed),
                stagnant_windows=0,
                deleted=False,
            )

        with self.storage.lock(self.episode_state):
            diagnostics = self.stagnation.record(
                state=self.episode_state,
                score=score,
                progressed=progressed,
            )

            if diagnostics.deleted:
                self.storage.savestate(self.episode_state).unlink(
                    missing_ok=True
                )
                self.mastery.clear(self.episode_state)
                self.stagnation.clear(self.episode_state)

            return diagnostics

    @property
    def required_successes(self) -> int:
        return self.mastery.required_successes

    def _practice_or_full_run(self) -> tuple[str, bytes | None]:
        practice = tuple(
            (state, self.storage.savestate(state))
            for state in self.states[1:]
            if self.storage.savestate(state).is_file()
        )

        if practice and random.random() >= self.full_run_probability:
            self.episode_state, savestate = random.choice(practice)
            return self.episode_state, savestate.read_bytes()

        self.full_run = True
        self.episode_state = self.states[0]
        return self.episode_state, None

    def _record_success(self, state: str) -> tuple[int, bool]:
        if self.storage.completed(state):
            return self.required_successes, True

        successes, completed = self.mastery.record(state)
        self.stagnation.clear(state)

        if completed:
            self.storage.complete(state)

        return successes, completed

    def _require_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown curriculum state: {state}")


def _is_stagnant(
    recent_median: float,
    best_median: float,
    trend: float,
    progressed: np.ndarray,
    scale: float,
) -> bool:
    score_not_improving = recent_median <= best_median + scale
    trend_not_improving = trend <= scale / max(len(progressed), 1)
    no_progress = not np.any(progressed)

    return score_not_improving and trend_not_improving and no_progress


def _robust_scale(values: np.ndarray) -> float:
    median = np.median(values)
    median_absolute_deviation = np.median(np.abs(values - median))

    if median_absolute_deviation > 0.0:
        return float(1.4826 * median_absolute_deviation)

    differences = np.abs(np.diff(values))
    nonzero_differences = differences[differences > 0.0]

    if nonzero_differences.size:
        return float(np.median(nonzero_differences))

    magnitude = max(float(np.max(np.abs(values))), 1.0)
    return float(np.finfo(np.float64).eps * magnitude)


def _theil_sen_trend(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0

    slopes = np.asarray(
        [
            (values[right] - values[left]) / (right - left)
            for left in range(len(values) - 1)
            for right in range(left + 1, len(values))
        ],
        dtype=np.float64,
    )

    return float(np.median(slopes))


def _finite_or_default(value: float, default: float) -> float:
    return float(value) if np.isfinite(value) else float(default)