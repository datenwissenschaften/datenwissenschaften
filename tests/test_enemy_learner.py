from pathlib import Path
from types import SimpleNamespace

import numpy as np
from datenwissenschaften.helpers.position import Position
from datenwissenschaften.vision import enemy_learner
from datenwissenschaften.vision.enemy_learner import EnemyLearner


def test_hit_signal_learns_and_persists_enemy_crop(tmp_path: Path, monkeypatch):
    runtime = SimpleNamespace(cache_dir=tmp_path, game="Game", savestate="Level1")
    monkeypatch.setattr(enemy_learner, "get_runtime", lambda: runtime)
    monkeypatch.setattr(enemy_learner, "publish_metadata", lambda *_args, **_kwargs: None)
    learner = EnemyLearner("Explore")
    before = np.zeros((96, 96, 3), dtype=np.uint8)
    collision = before.copy()
    collision[40:52, 45:57] = np.asarray([255, 80, 20], dtype=np.uint8)

    learner.observe(before, Position(128, 128), hit=False)
    observation = learner.observe(collision, Position(128, 128), hit=True)

    assert observation.learned_enemy_ids
    learned = list((tmp_path / "learned_enemies" / "Game" / "Level1" / "Explore").glob("*.png"))
    assert learned


def test_hit_signal_only_learns_on_rising_edge(tmp_path: Path, monkeypatch):
    runtime = SimpleNamespace(cache_dir=tmp_path, game="Game", savestate="Level1")
    monkeypatch.setattr(enemy_learner, "get_runtime", lambda: runtime)
    monkeypatch.setattr(enemy_learner, "publish_metadata", lambda *_args, **_kwargs: None)
    learner = EnemyLearner("Explore")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    learner.observe(frame, Position(128, 128), hit=False)
    first = learner.observe(frame, Position(128, 128), hit=True)
    second = learner.observe(frame, Position(128, 128), hit=True)

    assert first.learned_enemy_ids
    assert second.learned_enemy_ids == ()
