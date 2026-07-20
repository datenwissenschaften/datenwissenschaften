from dataclasses import dataclass, field


def _require_won(info: dict) -> bool:
    if "won" not in info:
        raise ValueError("Episode info must include a 'won' boolean field.")
    if not isinstance(info["won"], bool):
        raise ValueError("Episode info field 'won' must be a boolean.")
    return info["won"]


@dataclass
class EpisodeRecord:
    env_index: int
    episode_index: int
    bk2_path: str = field(init=False)
    won: bool = field(init=False)
    step_count: int = field(init=False)
    started_from_initial_savestate: bool | None = field(init=False)
    score: float = field(init=False)
    curriculum_state: str | None = field(init=False)
    episode_start_state: str | None = field(init=False)
    curriculum_succeeded: bool = field(init=False)
    curriculum_mastered: bool = field(init=False)

    def __post_init__(self) -> None:
        self.bk2_path = ""
        self.won = False
        self.step_count = 0
        self.started_from_initial_savestate = None
        self.score = 0.0
        self.curriculum_state = None
        self.episode_start_state = None
        self.curriculum_succeeded = False
        self.curriculum_mastered = False

    def add_step(self, info: dict, reward: float | None = None) -> None:
        if self.step_count == 0:
            bk2_path = info.get("episode_bk2_path")
            if isinstance(bk2_path, str) and bk2_path:
                self.bk2_path = bk2_path
            started_from_initial = info.get("started_from_initial_savestate")
            if isinstance(started_from_initial, bool):
                self.started_from_initial_savestate = started_from_initial
            curriculum_state = info.get("curriculum_state")
            if curriculum_state:
                self.curriculum_state = str(curriculum_state)
            episode_start_state = info.get("episode_start_state")
            if episode_start_state:
                self.episode_start_state = str(episode_start_state)
        self.step_count += 1
        self.score += float(info.get("extrinsic_reward", reward or 0.0))
        monitor_episode = info.get("episode")
        if isinstance(monitor_episode, dict) and monitor_episode.get("r") is not None:
            self.score = float(monitor_episode["r"])
        if _require_won(info) and not self.won:
            self.won = True
        if info.get("curriculum_succeeded") is True:
            self.curriculum_succeeded = True
        if info.get("curriculum_mastered") is True:
            self.curriculum_mastered = True

    def clone(self) -> "EpisodeRecord":
        episode = EpisodeRecord(self.env_index, self.episode_index)
        episode.bk2_path = self.bk2_path
        episode.won = self.won
        episode.step_count = self.step_count
        episode.started_from_initial_savestate = self.started_from_initial_savestate
        episode.score = self.score
        episode.curriculum_state = self.curriculum_state
        episode.episode_start_state = self.episode_start_state
        episode.curriculum_succeeded = self.curriculum_succeeded
        episode.curriculum_mastered = self.curriculum_mastered
        return episode
