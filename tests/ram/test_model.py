from dataclasses import make_dataclass

import numpy as np

from datenwissenschaften.ram.model import RamInfo, ram_word, signed_ram


def test_decodes_signed_bytes_and_words() -> None:
    ram_type = make_dataclass(
        "TestRam",
        (
            ("speed", int, signed_ram(0x10)),
            ("camera", int, ram_word(0x20)),
        ),
        bases=(RamInfo,),
    )
    raw_ram = np.zeros(0x22, dtype=np.uint8)
    raw_ram[0x10] = 0xFE
    raw_ram[0x20:0x22] = (0x34, 0x12)

    value = ram_type.from_ram(raw_ram)

    assert value.speed == -2
    assert value.camera == 0x1234
    assert value.features() == [-2 / 128.0, 0x1234 / 65535.0]
