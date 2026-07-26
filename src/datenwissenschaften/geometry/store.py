import sqlite3
from pathlib import Path

EMPTY = 0
SOLID = 1
HAZARDOUS = 2
UNKNOWN = 3


class TileStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS tiles (x INTEGER, y INTEGER, status INTEGER, PRIMARY KEY (x, y))"
        )
        self.tiles: dict[tuple[int, int], int] = {}
        self.refresh()

    def refresh(self) -> None:
        rows = self.connection.execute("SELECT x, y, status FROM tiles")
        for x, y, status in rows:
            coordinate = int(x), int(y)
            self.tiles[coordinate] = max(self.tiles.get(coordinate, EMPTY), int(status))

    def status(self, coordinate: tuple[int, int]) -> int:
        return self.tiles.get(coordinate, UNKNOWN)

    def remember(self, observations: dict[tuple[int, int], int]) -> None:
        updates = [
            (x, y, status)
            for (x, y), status in observations.items()
            if status > self.tiles.get((x, y), -1) and status != UNKNOWN
        ]
        if not updates:
            return
        self.connection.executemany(
            """
            INSERT INTO tiles (x, y, status) VALUES (?, ?, ?)
            ON CONFLICT (x, y) DO UPDATE SET status = MAX(status, excluded.status)
            """,
            updates,
        )
        for x, y, _ in updates:
            row = self.connection.execute("SELECT status FROM tiles WHERE x = ? AND y = ?", (x, y)).fetchone()
            if row is None:
                raise RuntimeError(f"Tile write failed at ({x}, {y})")
            self.tiles[(x, y)] = int(row[0])
