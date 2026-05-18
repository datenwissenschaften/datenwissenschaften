from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetroArenaPaths:
    roms_path: Path
    models_dir: Path
    working_dir: Path
    record_dir: Path
