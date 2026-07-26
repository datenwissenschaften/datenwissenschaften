from dataclasses import dataclass

from datenwissenschaften.ram.model import RamInfo, ram, ram_array


def genesis_uint(raw_bytes: list[int]) -> int:
    ordered = bytearray()
    for index in range(0, len(raw_bytes), 2):
        ordered.extend((raw_bytes[index + 1], raw_bytes[index]))
    return int.from_bytes(ordered, byteorder="big")


@dataclass(frozen=True, slots=True)
class AirstrikerRam(RamInfo):
    score_bytes: list[int] = ram_array(0x024E, 4)
    lives_bytes: list[int] = ram_array(0x025A, 2)
    game_over_bytes: list[int] = ram_array(0x0266, 2)
    position_x: int = ram(0)
    position_y: int = ram(1)
    screen_x: int = ram(2)
    screen_y: int = ram(3)

    @property
    def score(self) -> int:
        return genesis_uint(self.score_bytes)

    @property
    def lives(self) -> int:
        return genesis_uint(self.lives_bytes)

    @property
    def game_over(self) -> bool:
        return genesis_uint(self.game_over_bytes) == 1 and self.lives == 0
