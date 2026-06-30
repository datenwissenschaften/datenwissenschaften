from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_model_state(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(payload, file)
    temporary_path.replace(path)


def load_model_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    zip_path = Path(f"{path}.zip")
    if not path.exists() and zip_path.exists():
        path = zip_path
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid NEAT model payload in {path}")
    return payload
