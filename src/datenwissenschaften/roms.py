import os
import subprocess
import sys


def import_roms(roms_dir: str | None = None) -> None:
    roms_dir = roms_dir or _required_env("RETRO_SPEEDLAB_ROM_PATH")
    subprocess.run(
        [sys.executable, "-m", "stable_retro.import", roms_dir],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} must be set.")
    return value
