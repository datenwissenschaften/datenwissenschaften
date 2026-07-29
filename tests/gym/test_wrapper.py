from dataclasses import make_dataclass

import pytest

from datenwissenschaften.gym.wrapper import _observation_space
from datenwissenschaften.ram.model import REQUIRED_DQN_RAM_FIELDS, RamInfo, ram


def ram_info(fields: tuple[str, ...]) -> type[RamInfo]:
    definitions = [(name, int, ram(address)) for address, name in enumerate(fields)]
    return make_dataclass("TestRam", definitions, bases=(RamInfo,), slots=True)


def test_accepts_all_required_dqn_ram_fields() -> None:
    space = _observation_space(ram_info(REQUIRED_DQN_RAM_FIELDS), ())

    assert space.shape == (9,)


def test_rejects_missing_dqn_ram_fields() -> None:
    with pytest.raises(ValueError, match=r"ram\.player_y"):
        _observation_space(ram_info(REQUIRED_DQN_RAM_FIELDS[:-1]), ())
