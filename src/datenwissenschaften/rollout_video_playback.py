from __future__ import annotations

import argparse
import sys

import stable_retro as retro
from stable_retro.scripts.playback_movie import _play as play_movie

from datenwissenschaften.roms import import_roms


def main(argv: list[str] | None = None) -> None:
    """Import configured ROMs before replaying a stable-retro movie."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--roms-dir")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("movies", nargs="+")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    args.lossless = None
    args.no_video = False
    args.info_dict = False
    args.npy_actions = False
    args.viewer = None
    args.ending = None

    import_roms(args.roms_dir)
    original_make = retro.make

    def make_headless(*args, **kwargs):
        kwargs["render_mode"] = "rgb_array"
        return original_make(*args, **kwargs)

    retro.make = make_headless
    try:
        for movie in args.movies:
            play_movie(movie, args, None)
    finally:
        retro.make = original_make


if __name__ == "__main__":
    main()
