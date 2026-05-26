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


def _number_info(info: dict, *keys: str) -> int | float | None:
    for key in keys:
        if key in info:
            return _number_value(info[key])

    state = info.get("state")
    if isinstance(state, dict):
        for key in keys:
            if key in state:
                return _number_value(state[key])

    return None


def _number_value(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return None
    return None


@dataclass
class EpisodeRecord:
    env_index: int
    episode_index: int
    bk2_path: str = field(init=False)
    won: bool = field(init=False)
    time_until_won: int = field(init=False)
    max_x: int | float | None = field(init=False)

    def __post_init__(self) -> None:
        self.bk2_path = ""
        self.won = False
        self.time_until_won = 0
        self.max_x = None

    def add_step(self, info: dict) -> None:
        step_max_x = _number_info(info, "max_x")
        if step_max_x is not None and (self.max_x is None or step_max_x > self.max_x):
            self.max_x = step_max_x

        if self.won:
            return

        self.time_until_won += 1
        self.won = _bool_info(info, "won")

    def clone(self) -> "EpisodeRecord":
        episode = EpisodeRecord(self.env_index, self.episode_index)
        episode.bk2_path = self.bk2_path
        episode.won = self.won
        episode.time_until_won = self.time_until_won
        episode.max_x = self.max_x
        return episode
