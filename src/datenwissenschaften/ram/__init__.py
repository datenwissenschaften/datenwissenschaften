from dataclasses import field, fields
from typing import Any, Self, final


def ram(address: int):
    return field(default=0, metadata={"address": address})


class RamInfo:
    @classmethod
    @final
    def ram_map(cls) -> dict[str, int]:
        return {f.name: int(f.metadata["address"]) for f in fields(cls)}

    @classmethod
    @final
    def from_ram(cls, raw_ram: Any) -> Self:
        return cls(**{name: int(raw_ram[address]) for name, address in cls.ram_map().items()})
