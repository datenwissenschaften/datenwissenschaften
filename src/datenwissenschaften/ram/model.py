from dataclasses import field, fields
from typing import Any, Self, final

REQUIRED_RAM_FIELDS = (
    "screen_x",
    "screen_y",
    "player_x",
    "player_y",
)


def ram(address: int) -> Any:
    return field(default=0, metadata={"address": address, "length": 1})


def signed_ram(address: int) -> Any:
    return field(default=0, metadata={"address": address, "length": 1, "signed": True})


def ram_word(address: int) -> Any:
    return field(default=0, metadata={"address": address, "length": 2, "word": True})


def ram_array(address: int, length: int) -> Any:
    if length < 1:
        raise ValueError("length must be positive.")

    return field(
        default_factory=lambda: [0] * length,
        metadata={"address": address, "length": length, "array": True},
    )


class RamInfo:
    @classmethod
    @final
    def ram_map(cls) -> dict[str, tuple[int, int]]:
        return {f.name: (int(f.metadata["address"]), int(f.metadata["length"])) for f in fields(cls)}

    @classmethod
    @final
    def feature_size(cls) -> int:
        return sum(1 if f.metadata.get("word") else int(f.metadata["length"]) for f in fields(cls))

    @classmethod
    @final
    def from_ram(cls, raw_ram: Any) -> Self:
        values = {}
        ram_fields = {item.name: item for item in fields(cls)}

        for name, (address, length) in cls.ram_map().items():
            ram_field = ram_fields[name]
            if ram_field.metadata.get("word"):
                values[name] = int(raw_ram[address]) | int(raw_ram[address + 1]) << 8
            elif length == 1:
                value = int(raw_ram[address])
                values[name] = value - 256 if ram_field.metadata.get("signed") and value >= 128 else value
            else:
                values[name] = [int(raw_ram[address + offset]) for offset in range(length)]

        return cls(**values)

    @final
    def to_dict(self) -> dict[str, int | list[int]]:
        values: dict[str, int | list[int]] = {}

        for name in self.ram_map():
            value = getattr(self, name)
            values[name] = list(value) if isinstance(value, list) else int(value)

        return values

    def features(self) -> list[float]:
        result: list[float] = []

        for f in fields(self):
            value = getattr(self, f.name)
            scale = 65535.0 if f.metadata.get("word") else 128.0 if f.metadata.get("signed") else 255.0

            if isinstance(value, list):
                result.extend(float(item) / scale for item in value)
            else:
                result.append(float(value) / scale)

        return result
