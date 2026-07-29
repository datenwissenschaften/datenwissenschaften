import tempfile
from pathlib import Path
from typing import Any


def atomic_save(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(".zip")
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        model.save(temporary)
        temporary.with_suffix(".zip").replace(target)


def atomic_save_replay_buffer(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        model.save_replay_buffer(temporary)
        temporary.replace(path)
