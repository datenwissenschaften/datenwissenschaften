from dataclasses import field, fields
from typing import Any, Self, final


def ram(address: int):
    return field(default=0, metadata={"address": address, "length": 1})


def ram_array(address: int, length: int):
    if length < 1:
        raise ValueError("length must be positive.")

    return field(
        default_factory=lambda: [0] * length,
        metadata={"address": address, "length": length},
    )


class RamInfo:
    @classmethod
    @final
    def ram_map(cls) -> dict[str, tuple[int, int]]:
        return {f.name: (int(f.metadata["address"]), int(f.metadata["length"])) for f in fields(cls)}

    @classmethod
    @final
    def from_ram(cls, raw_ram: Any) -> Self:
        values = {}

        for name, (address, length) in cls.ram_map().items():
            if length == 1:
                values[name] = int(raw_ram[address])
            else:
                values[name] = [int(raw_ram[address + offset]) for offset in range(length)]

        return cls(**values)

    def features(self) -> list[float]:
        result = []

        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, list):
                result.extend(float(item) / 255.0 for item in value)
            else:
                result.append(float(value) / 255.0)

        return result
