import fcntl
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class CurriculumStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def savestate(self, state: str) -> Path:
        return self.root / f"{state}.state"

    def completed(self, state: str) -> bool:
        return (self.root / f"{state}.complete").is_file()

    def complete(self, state: str) -> None:
        self._write(self.root / f"{state}.complete", b"")

    def save(self, state: str, savestate: bytes) -> None:
        self._write(self.savestate(state), savestate)

    @contextmanager
    def lock(self, state: str) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f".{state}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / path.name
            temporary.write_bytes(content)
            temporary.replace(path)
