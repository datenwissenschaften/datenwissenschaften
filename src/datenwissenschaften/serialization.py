from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_json_value(value: Any) -> Any:
    """Convert nested application values to JSON-compatible primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_value(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return str(value)
