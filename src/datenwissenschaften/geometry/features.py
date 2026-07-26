from datenwissenschaften.geometry.player import PlayerGeometry
from datenwissenschaften.geometry.store import EMPTY, SOLID, UNKNOWN, TileStore

FEATURE_COUNT = 13
LOOKAHEAD_TILES = 8
FRAMES_TO_GAP_LIMIT = 30.0


def geometry_features(store: TileStore, player: PlayerGeometry, tile_size: int, horizontal_speed: float) -> list[float]:
    half_tile = tile_size / 2
    distances = tuple(index * half_tile for index in range(LOOKAHEAD_TILES * 2 + 1))
    probe_y = player.bottom + half_tile
    statuses = tuple(
        _status(store, player.center_x + player.direction * distance, probe_y, tile_size) for distance in distances
    )
    gap_index = next((index for index, status in enumerate(statuses) if status == EMPTY), len(statuses) - 1)
    landing_index = next(
        (index for index in range(gap_index, len(statuses)) if statuses[index] == SOLID),
        len(statuses) - 1,
    )
    gap_end = next(
        (index for index in range(gap_index, len(statuses)) if statuses[index] != EMPTY),
        len(statuses) - 1,
    )
    landing_end = next(
        (index for index in range(landing_index, len(statuses)) if statuses[index] != SOLID),
        len(statuses) - 1,
    )
    horizon = distances[-1]
    gap_distance = distances[gap_index]
    samples = tuple(statuses[index] / UNKNOWN for index in (0, 2, 4, 6, 8))
    derived = (
        gap_distance / horizon,
        (distances[gap_end] - gap_distance) / horizon,
        distances[landing_index] / horizon,
        (distances[landing_end] - distances[landing_index]) / horizon,
        min(gap_distance / max(abs(horizontal_speed), 1.0) / FRAMES_TO_GAP_LIMIT, 1.0),
    )
    front_x = player.center_x + player.direction * tile_size
    obstacles = (
        _status(store, front_x, player.bottom - half_tile, tile_size) / UNKNOWN,
        _status(store, front_x, player.world_y + half_tile, tile_size) / UNKNOWN,
        _status(store, player.center_x, player.world_y - half_tile, tile_size) / UNKNOWN,
    )
    return [*samples, *derived, *obstacles]


def _status(store: TileStore, world_x: float, world_y: float, tile_size: int) -> int:
    return store.status((int(world_x // tile_size), int(world_y // tile_size)))
