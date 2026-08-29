from pathlib import Path

from datenwissenschaften.states.target_memory import TargetMemory


def test_remember_keeps_the_first_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "FindDispenser.json"

    TargetMemory(path).remember((800.0, 1579.0))
    memory = TargetMemory(path)
    memory.remember((12.0, 34.0))

    assert memory.coordinates == (800.0, 1579.0)
    assert TargetMemory(path).coordinates == (800.0, 1579.0)
