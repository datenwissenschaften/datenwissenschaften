import fcntl
from pathlib import Path


class EpisodeCounter:
    def __init__(self, path: Path) -> None:
        self.path: Path = path

    def next_episode(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a+", encoding="utf-8") as counter_file:
            fcntl.flock(counter_file.fileno(), fcntl.LOCK_EX)
            counter_file.seek(0)
            content = counter_file.read().strip()
            episode_number = int(content) + 1 if content else 1
            counter_file.seek(0)
            counter_file.truncate()
            counter_file.write(str(episode_number))
            counter_file.flush()
            fcntl.flock(counter_file.fileno(), fcntl.LOCK_UN)

        return episode_number
