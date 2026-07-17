from pathlib import Path
from types import SimpleNamespace

import cv2
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
    collision[40:52, 54:66] = np.asarray([255, 80, 20], dtype=np.uint8)
    following_hit_frame = collision.copy()
    following_hit_frame[35:47, 60:72] = np.asarray([20, 180, 255], dtype=np.uint8)

    learner.observe(before, Position(128, 128), hit=False)
    observation = learner.observe(collision, Position(128, 128), hit=True)
    consecutive = learner.observe(following_hit_frame, Position(128, 128), hit=True)

    assert observation.learned_enemy_ids
    assert consecutive.learned_enemy_ids == ()
    learned = list((tmp_path / "learned_enemies" / "Game").glob("*.png"))
    assert len(learned) == len(observation.learned_enemy_ids)
    sprite = cv2.imread(str(learned[0]), cv2.IMREAD_UNCHANGED)
    assert sprite.shape[2] == 4
    assert 16 <= np.count_nonzero(sprite[..., 3]) < sprite[..., 3].size


def test_hit_signal_without_motion_does_not_learn_background(tmp_path: Path, monkeypatch):
    runtime = SimpleNamespace(cache_dir=tmp_path, game="Game", savestate="Level1")
    monkeypatch.setattr(enemy_learner, "get_runtime", lambda: runtime)
    monkeypatch.setattr(enemy_learner, "publish_metadata", lambda *_args, **_kwargs: None)
    learner = EnemyLearner("Explore")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    learner.observe(frame, Position(128, 128), hit=False)
    first = learner.observe(frame, Position(128, 128), hit=True)
    second = learner.observe(frame, Position(128, 128), hit=True)

    assert first.learned_enemy_ids == ()
    assert second.learned_enemy_ids == ()


def test_templates_are_shared_across_savestates_for_the_game(tmp_path: Path, monkeypatch):
    runtime = SimpleNamespace(cache_dir=tmp_path, game="Game", savestate="Level1")
    monkeypatch.setattr(enemy_learner, "get_runtime", lambda: runtime)
    monkeypatch.setattr(enemy_learner, "publish_metadata", lambda *_args, **_kwargs: None)
    learner = EnemyLearner("Explore")
    before = np.zeros((96, 96, 3), dtype=np.uint8)
    collision = before.copy()
    collision[40:52, 54:66] = 255

    learner.observe(before, Position(128, 128), hit=False)
    learned = learner.observe(collision, Position(128, 128), hit=True).learned_enemy_ids
    assert learned

    runtime.savestate = "Level2"
    learner.observe(before, Position(128, 128), hit=False)
    assert set(learner.templates) == set(learned)
    assert list((tmp_path / "learned_enemies" / "Game").glob("*.png"))
