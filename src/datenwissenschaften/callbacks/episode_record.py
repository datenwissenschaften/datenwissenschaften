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
    time_until_won: int | None = field(init=False)

    def __post_init__(self) -> None:
        self.bk2_path = ""
        self.won = False
        self.step_count = 0
        self.time_until_won = None

    def add_step(self, info: dict) -> None:
        self.step_count += 1
        if _require_won(info) and not self.won:
            self.won = True
            self.time_until_won = self.step_count

    def clone(self) -> "EpisodeRecord":
        episode = EpisodeRecord(self.env_index, self.episode_index)
        episode.bk2_path = self.bk2_path
        episode.won = self.won
        episode.step_count = self.step_count
        episode.time_until_won = self.time_until_won
        return episode
