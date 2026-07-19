from __future__ import annotations

import sys

from stable_retro.scripts.playback_movie import main as playback_main

from datenwissenschaften.roms import import_roms


def main(argv: list[str] | None = None) -> None:
    """Import configured ROMs before replaying a stable-retro movie."""
    import_roms()
    playback_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
