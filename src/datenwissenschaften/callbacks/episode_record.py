from dataclasses import dataclass, field


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
    progress: int | float | None = field(init=False)

    def __post_init__(self) -> None:
        self.bk2_path = ""
        self.progress = None

    def add_step(self, info: dict) -> None:
        step_progress = _number_info(info, "progress")
        if step_progress is not None and (self.progress is None or step_progress > self.progress):
            self.progress = step_progress

    def clone(self) -> "EpisodeRecord":
        episode = EpisodeRecord(self.env_index, self.episode_index)
        episode.bk2_path = self.bk2_path
        episode.progress = self.progress
        return episode
