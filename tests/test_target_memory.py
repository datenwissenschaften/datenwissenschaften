from types import SimpleNamespace

from datenwissenschaften.states.target_memory import TargetMemory


def test_reset_all_deletes_persisted_and_live_target_memories(monkeypatch):
    deleted = []
    memory = SimpleNamespace(
        _store=SimpleNamespace(delete=lambda *scope: deleted.append(scope)),
        _scope=("target-memory", "Game", "Level", "Target"),
        coordinates=(12.0, 34.0),
    )
    monkeypatch.setattr(TargetMemory, "_registry", {"Target": memory})

    TargetMemory.reset_all()

    assert deleted == [("target-memory", "Game", "Level", "Target")]
    assert memory.coordinates is None
