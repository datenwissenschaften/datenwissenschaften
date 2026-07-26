import subprocess
import sys
from pathlib import Path

from loguru import logger


def import_roms(roms_dir: str | Path) -> None:
    logger.info("Importing ROMs from {}", Path(roms_dir).resolve())
    subprocess.run(
        [sys.executable, "-m", "stable_retro.import", str(roms_dir)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
