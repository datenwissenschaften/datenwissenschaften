from dataclasses import dataclass, field


def _bool_info(info: dict, *keys: str) -> bool:
    for key in keys:
        if key in info:
            return bool(info[key])

    state = info.get("state")
    if isinstance(state, dict):
        for key in keys:
            if key in state:
                return bool(state[key])

    return False


@dataclass
class EpisodeRecord:
    env_index: int
    episode_index: int
    bk2_path: str = field(init=False)
    won: bool = field(init=False)
    time_until_won: int = field(init=False)

    def __post_init__(self) -> None:
        self.bk2_path = ""
        self.won = False
        self.time_until_won = 0

    def add_step(self, info: dict) -> None:
        if self.won:
            return

        self.time_until_won += 1
        self.won = _bool_info(info, "won")

    def clone(self) -> "EpisodeRecord":
        episode = EpisodeRecord(self.env_index, self.episode_index)
        episode.bk2_path = self.bk2_path
        episode.won = self.won
        episode.time_until_won = self.time_until_won
        return episode
