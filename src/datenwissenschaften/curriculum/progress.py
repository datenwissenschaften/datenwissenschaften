import random
from pathlib import Path

from datenwissenschaften.curriculum.mastery import StageMastery
from datenwissenschaften.curriculum.stagnation import ScoreStagnation
from datenwissenschaften.curriculum.storage import CurriculumStorage


class SavestateCurriculum:
    def __init__(
        self,
        root: Path,
        states: tuple[str, ...],
        required_successes: int,
        full_run_probability: float,
    ) -> None:
        if not states:
            raise ValueError("Curriculum requires at least one state")
        if not 0.0 <= full_run_probability <= 1.0:
            raise ValueError("full_run_probability must be between zero and one")
        self.root = root
        self.states = states
        self.full_run_probability = full_run_probability
        self.episode_state = states[0]
        self.recorded = False
        self.training = True
        self.full_run = False
        self.mastery = StageMastery(root, required_successes)
        self.stagnation: ScoreStagnation = ScoreStagnation(root)
        self.storage = CurriculumStorage(root)

    def start(self) -> tuple[str, bytes | None]:
        self.recorded = False
        self.full_run = False
        target_index = next(
            (index for index, state in enumerate(self.states) if not self.storage.completed(state)),
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

    def transition(self, previous: str, current: str, savestate: bytes) -> tuple[int, bool]:
        self._require_state(current)
        if not self.training or self.recorded or previous != self.episode_state:
            return 0, False
        with self.storage.lock(previous):
            self.storage.save(current, savestate)
            successes, completed = self._record_success(previous)
        self.recorded = True
        return successes, completed

    def victory(self, state: str) -> tuple[int, bool]:
        self._require_state(state)
        if not self.training or self.recorded or state != self.episode_state:
            return 0, False
        with self.storage.lock(state):
            successes, completed = self._record_success(state)
        self.recorded = True
        return successes, completed

    def record_attempt(self, score: float) -> tuple[int, bool]:
        if not self.training or self.recorded or self.episode_state == self.states[0]:
            return 0, False
        with self.storage.lock(self.episode_state):
            episodes, deleted = self.stagnation.record(self.episode_state, score)
            if deleted:
                self.mastery.clear(self.episode_state)
            return episodes, deleted

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
        if completed:
            self.storage.complete(state)
            self.stagnation.clear(state)
        return successes, completed

    def _require_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown curriculum state: {state}")
