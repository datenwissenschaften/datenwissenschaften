from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from datenwissenschaften.vision import enemy_learner
from datenwissenschaften.vision.enemy_learner import EnemyLearner

ACTOR_COLOR = np.asarray([30, 220, 80], dtype=np.uint8)
GOOMBA_COLOR = np.asarray([180, 60, 20], dtype=np.uint8)


def _configure(tmp_path: Path, monkeypatch) -> EnemyLearner:
    runtime = SimpleNamespace(cache_dir=tmp_path, game="Game", savestate="Level1")
    monkeypatch.setattr(enemy_learner, "get_runtime", lambda: runtime)
    monkeypatch.setattr(enemy_learner, "publish_metadata", lambda *_args, **_kwargs: None)
    return EnemyLearner("Explore")


def _frame(actor_x: int, extras=()) -> np.ndarray:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[48:60, actor_x : actor_x + 12] = ACTOR_COLOR
    for y1, y2, x1, x2, color in extras:
        frame[y1:y2, x1:x2] = color
    return frame


def _prime_actor(learner: EnemyLearner) -> int:
    actor_x = 0
    for actor_x in range(18, 36, 2):
        learner.observe(_frame(actor_x), hit=False)
    assert learner.actor_confident
    assert learner.actor_center is not None
    return actor_x


def test_visual_actor_is_learned_before_touched_enemy(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner) + 2
    collision = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])

    observation = learner.observe(collision, hit=True)
    consecutive = learner.observe(collision, hit=True)

    assert observation.learned_enemy_ids
    assert consecutive.learned_enemy_ids == ()
    learned = list((tmp_path / "learned_enemies" / "Game").glob("*.png"))
    assert len(learned) == len(observation.learned_enemy_ids)
    sprite = cv2.imread(str(learned[0]), cv2.IMREAD_UNCHANGED)
    assert sprite.shape[2] == 4
    assert 16 <= np.count_nonzero(sprite[..., 3]) < sprite[..., 3].size


def test_hit_does_not_learn_until_actor_is_visually_confident(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    before = _frame(30)
    collision = _frame(32, [(48, 60, 45, 57, GOOMBA_COLOR)])

    learner.observe(before, hit=False)
    observation = learner.observe(collision, hit=True)

    assert not learner.actor_confident
    assert observation.learned_enemy_ids == ()


def test_single_changed_enemy_pixel_expands_to_complete_sprite(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner)
    enemy_x = actor_x + 13
    before = _frame(actor_x, [(48, 60, enemy_x, enemy_x + 12, GOOMBA_COLOR)])
    learner.observe(before, hit=False)
    collision = before.copy()
    collision[54, enemy_x + 6] = np.asarray([255, 60, 20], dtype=np.uint8)

    learned_ids = learner.observe(collision, hit=True).learned_enemy_ids

    assert learned_ids
    learned = cv2.imread(
        str(next((tmp_path / "learned_enemies" / "Game").glob("*.png"))),
        cv2.IMREAD_UNCHANGED,
    )
    assert cv2.countNonZero(learned[..., 3]) >= 100


def test_cloud_and_question_block_are_not_learned_instead_of_goomba(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner) + 2
    collision = _frame(
        actor_x,
        [
            (48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR),
            (22, 34, 38, 50, np.asarray([255, 180, 20], dtype=np.uint8)),
            (10, 20, 65, 88, np.asarray([220, 220, 255], dtype=np.uint8)),
        ],
    )

    learned_ids = learner.observe(collision, hit=True).learned_enemy_ids

    assert len(learned_ids) == 1
    learned = cv2.imread(
        str(next((tmp_path / "learned_enemies" / "Game").glob("*.png"))),
        cv2.IMREAD_UNCHANGED,
    )
    foreground_bgr = learned[..., :3][learned[..., 3] > 0]
    expected_bgr = GOOMBA_COLOR[::-1]
    assert np.linalg.norm(np.median(foreground_bgr, axis=0) - expected_bgr) < 10


def test_stationary_animation_cannot_become_actor(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    for step, actor_x in enumerate(range(18, 38, 2)):
        frame = _frame(actor_x, [(22, 34, 55, 67, np.asarray([180 + step % 2 * 40, 120, 20], dtype=np.uint8))])
        learner.observe(frame, hit=False)

    assert learner.actor_confident
    assert learner.actor_center is not None
    assert abs(learner.actor_center[1] - 54) <= 2


def test_templates_are_shared_across_savestates_for_the_game(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    runtime = enemy_learner.get_runtime()
    actor_x = _prime_actor(learner) + 2
    collision = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])
    learned = learner.observe(collision, hit=True).learned_enemy_ids
    assert learned

    runtime.savestate = "Level2"
    learner.observe(_frame(actor_x), hit=False)
    assert set(learner.templates) == set(learned)
    assert list((tmp_path / "learned_enemies" / "Game").glob("*.png"))


def test_unverified_legacy_candidates_are_quarantined(tmp_path: Path, monkeypatch):
    root = tmp_path / "learned_enemies" / "Game"
    root.mkdir(parents=True)
    legacy = np.zeros((12, 12, 4), dtype=np.uint8)
    legacy[2:10, 2:10] = 255
    assert cv2.imwrite(str(root / "cloud.png"), legacy)

    learner = _configure(tmp_path, monkeypatch)
    learner.observe(_frame(20), hit=False)

    assert not list(root.glob("*.png"))
    assert (root / "cloud.png.unverified").is_file()
