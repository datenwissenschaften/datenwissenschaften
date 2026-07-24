from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften import model


class FakeRedisStore:
    values = {}
    deleted = []

    def __init__(self, redis_url, *, key_prefix="datenwissenschaften"):
        self.key_prefix = key_prefix

    def get(self, *parts, default=None):
        return self.values.get((self.key_prefix, *parts), default)

    def set(self, *parts, value):
        self.values[(self.key_prefix, *parts)] = value

    def delete(self, *parts):
        self.deleted.append((self.key_prefix, *parts))

    def delete_prefix(self, *parts):
        self.deleted.append((self.key_prefix, *parts, "*"))


def _config(root=Path(".")):
    return SimpleNamespace(
        paths=SimpleNamespace(
            models_dir=root / "models",
            record_dir=root / "recordings",
            cache_dir=root / "cache",
        ),
        training=SimpleNamespace(game="Game", game_identity="Game", fingerprint="fingerprint-a"),
        ui=SimpleNamespace(
            redis_url="redis://example",
            history_key_prefix="datenwissenschaften:history",
        ),
    )


def test_version_change_clears_artifacts_states_history_and_live_memory(monkeypatch, tmp_path: Path):
    FakeRedisStore.values = {
        ("datenwissenschaften", "engine-version", "Game"): "2.9.30",
        ("datenwissenschaften", "database-fingerprint", "Game"): "fingerprint-a",
    }
    FakeRedisStore.deleted = []
    environment_calls = []
    monkeypatch.setattr(model, "RedisStore", FakeRedisStore)
    venv = SimpleNamespace(env_method=lambda method: environment_calls.append(method))
    for path in (tmp_path / "models", tmp_path / "recordings", tmp_path / "cache"):
        path.mkdir()
        (path / "stale-artifact").write_text("stale", encoding="utf-8")

    changed = model.reset_for_training_change(
        _config(tmp_path),
        venv,
        config_path=tmp_path / "config.yaml",
        current_version="2.10.0",
    )

    assert changed is True
    assert all(
        path.is_dir() and not list(path.iterdir())
        for path in (tmp_path / "models", tmp_path / "recordings", tmp_path / "cache")
    )
    assert FakeRedisStore.deleted == [
        ("datenwissenschaften", "state", "Game", "*"),
        ("datenwissenschaften", "target-memory", "Game", "*"),
        ("datenwissenschaften", "active-savestate", "Game"),
        ("datenwissenschaften:history", "Game"),
    ]
    assert environment_calls == ["reset_training_memory"]
    assert FakeRedisStore.values[("datenwissenschaften", "engine-version", "Game")] == "2.10.0"
    assert FakeRedisStore.values[("datenwissenschaften", "database-fingerprint", "Game")] == "fingerprint-a"


def test_matching_version_keeps_existing_training_data(monkeypatch):
    FakeRedisStore.values = {
        ("datenwissenschaften", "engine-version", "Game"): "2.10.0",
        ("datenwissenschaften", "database-fingerprint", "Game"): "fingerprint-a",
    }
    FakeRedisStore.deleted = []
    monkeypatch.setattr(model, "RedisStore", FakeRedisStore)

    changed = model.reset_for_training_change(
        _config(),
        SimpleNamespace(),
        current_version="2.10.0",
    )

    assert changed is False
    assert FakeRedisStore.deleted == []


def test_patch_version_change_keeps_existing_training_data_and_records_version(monkeypatch):
    FakeRedisStore.values = {
        ("datenwissenschaften", "engine-version", "Game"): "2.10.3",
        ("datenwissenschaften", "database-fingerprint", "Game"): "fingerprint-a",
    }
    FakeRedisStore.deleted = []
    monkeypatch.setattr(model, "RedisStore", FakeRedisStore)

    changed = model.reset_for_training_change(
        _config(),
        SimpleNamespace(),
        current_version="2.10.4",
    )

    assert changed is False
    assert FakeRedisStore.deleted == []
    assert FakeRedisStore.values[("datenwissenschaften", "engine-version", "Game")] == "2.10.4"


def test_only_major_or_minor_version_changes_require_a_reset():
    assert model._major_or_minor_version_changed("2.10.3", "2.10.4") is False
    assert model._major_or_minor_version_changed("2.10.4", "2.11.0") is True
    assert model._major_or_minor_version_changed("2.10.4", "3.0.0") is True
    assert model._major_or_minor_version_changed(None, "2.10.4") is False
    assert model._major_or_minor_version_changed("DEVELOPMENT", "2.10.4") is False


def test_database_fingerprint_change_resets_on_the_same_engine_version(monkeypatch, tmp_path: Path):
    FakeRedisStore.values = {
        ("datenwissenschaften", "engine-version", "Game"): "2.10.0",
        ("datenwissenschaften", "database-fingerprint", "Game"): "fingerprint-a",
    }
    FakeRedisStore.deleted = []
    monkeypatch.setattr(model, "RedisStore", FakeRedisStore)
    config = _config(tmp_path)
    config.training.fingerprint = "fingerprint-b"

    changed = model.reset_for_training_change(
        config,
        SimpleNamespace(),
        config_path=tmp_path / "config.yaml",
        current_version="2.10.0",
    )

    assert changed is True
    assert FakeRedisStore.values[("datenwissenschaften", "database-fingerprint", "Game")] == "fingerprint-b"
