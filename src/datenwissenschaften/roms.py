import subprocess
import sys
from pathlib import Path

from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config


def import_roms(roms_dir: str | Path | None = None, *, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    roms_dir = roms_dir or load_config(config_path).paths.roms_path
    subprocess.run(
        [sys.executable, "-m", "stable_retro.import", str(roms_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
