import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from datenwissenschaften.geometry.features import geometry_features
from datenwissenschaften.geometry.player import PlayerGeometry
from datenwissenschaften.geometry.store import EMPTY, HAZARDOUS, SOLID, TileStore

BACKGROUND_OBSERVATIONS_REQUIRED = 8
BACKGROUND_SHARE_REQUIRED = 0.75
BACKGROUND_SIGNATURES_KEPT = 4
STATIONARY_FRAMES_REQUIRED = 4
HAZARD_DEATHS_REQUIRED = 2


class TileMemory:
    def __init__(self, path: Path, tile_size: int) -> None:
        if tile_size < 1:
            raise ValueError("Tile size must be positive")
        self.tile_size: int = tile_size
        self.store: TileStore = TileStore(path)
        self.signatures: dict[tuple[int, int], Counter[int]] = {}
        self.latest_signatures: dict[tuple[int, int], int] = {}
        self.death_counts: Counter[tuple[int, int]] = Counter()
        self.previous_player: PlayerGeometry | None = None
        self.stationary_frames: int = 0
        self.horizontal_speed: float = 0.0

    def reset(self, frame: np.ndarray, player: PlayerGeometry) -> None:
        self.store.refresh()
        self.previous_player = None
        self.stationary_frames = 0
        self.horizontal_speed = 0.0
        self.observe(frame, player, False)

    def observe(self, frame: np.ndarray, player: PlayerGeometry, dead: bool) -> None:
        _observe_background(frame, player, self.tile_size, self.signatures, self.latest_signatures)
        observations = _occupied_tiles(player, self.tile_size)
        observations = {coordinate: EMPTY for coordinate in observations}
        support = _tile(player.center_x, player.bottom + 1, self.tile_size)
        hazard = _tile(player.center_x, player.bottom - 1, self.tile_size)
        previous = self.previous_player
        falling = False
        if previous is not None:
            self.horizontal_speed = player.world_x - previous.world_x
            if abs(player.world_y - previous.world_y) <= 1:
                self.stationary_frames += 1
            else:
                self.stationary_frames = 0
            falling = player.world_y - previous.world_y > 1
            if falling:
                observations[support] = EMPTY
        if self.stationary_frames >= STATIONARY_FRAMES_REQUIRED and self._stable_background(support):
            observations[support] = SOLID
        if dead and not falling and self._stable_background(hazard):
            self.death_counts[hazard] += 1
            if self.death_counts[hazard] >= HAZARD_DEATHS_REQUIRED:
                observations[hazard] = HAZARDOUS
        self.store.remember(observations)
        self.previous_player = player

    def features(self, player: PlayerGeometry) -> list[float]:
        return geometry_features(self.store, player, self.tile_size, self.horizontal_speed)

    def _stable_background(self, coordinate: tuple[int, int]) -> bool:
        counts = self.signatures.get(coordinate)
        latest = self.latest_signatures.get(coordinate)
        if counts is None or latest is None:
            return False
        observations = sum(counts.values())
        return (
            observations >= BACKGROUND_OBSERVATIONS_REQUIRED
            and counts[latest] / observations >= BACKGROUND_SHARE_REQUIRED
        )


def _observe_background(
    frame: np.ndarray,
    player: PlayerGeometry,
    tile_size: int,
    signatures: dict[tuple[int, int], Counter[int]],
    latest: dict[tuple[int, int], int],
) -> None:
    viewport_x = player.world_x - player.screen_x
    viewport_y = player.world_y - player.screen_y
    height, width = frame.shape[:2]
    first_x = int(viewport_x // tile_size)
    first_y = int(viewport_y // tile_size)
    last_x = int((viewport_x + width - 1) // tile_size)
    last_y = int((viewport_y + height - 1) // tile_size)
    for tile_y in range(first_y, last_y + 1):
        for tile_x in range(first_x, last_x + 1):
            left = round(tile_x * tile_size - viewport_x)
            top = round(tile_y * tile_size - viewport_y)
            right = left + tile_size
            bottom = top + tile_size
            if left < 0 or top < 0 or right > width or bottom > height:
                continue
            signature = zlib.crc32(frame[top:bottom, left:right].tobytes())
            coordinate = tile_x, tile_y
            counts = signatures.setdefault(coordinate, Counter())
            counts[signature] += 1
            if len(counts) > BACKGROUND_SIGNATURES_KEPT:
                del counts[counts.most_common()[-1][0]]
            latest[coordinate] = signature


def _occupied_tiles(player: PlayerGeometry, tile_size: int) -> set[tuple[int, int]]:
    left = int(player.world_x // tile_size)
    right = int((player.world_x + player.width - 1) // tile_size)
    top = int(player.world_y // tile_size)
    bottom = int((player.world_y + player.height - 1) // tile_size)
    return {(x, y) for x in range(left, right + 1) for y in range(top, bottom + 1)}


def _tile(world_x: float, world_y: float, tile_size: int) -> tuple[int, int]:
    return int(world_x // tile_size), int(world_y // tile_size)
