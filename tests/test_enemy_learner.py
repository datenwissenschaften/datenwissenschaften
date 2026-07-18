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


def test_enemy_requires_two_independent_hit_events_before_learning(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner) + 2
    collision = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])

    provisional = learner.observe(collision, hit=True)
    consecutive = learner.observe(collision, hit=True)

    root = tmp_path / "learned_enemies" / "Game"
    assert not list(root.glob("*.png"))
    assert list((root / ".candidates").glob("*.png"))

    learner.observe(_frame(actor_x), hit=False)
    confirmed = learner.observe(collision, hit=True)

    assert provisional.learned_enemy_ids == ()
    assert consecutive.learned_enemy_ids == ()
    assert confirmed.learned_enemy_ids
    learned = list((tmp_path / "learned_enemies" / "Game").glob("*.png"))
    assert len(learned) == len(confirmed.learned_enemy_ids)
    sprite = cv2.imread(str(learned[0]), cv2.IMREAD_UNCHANGED)
    assert sprite.shape[2] == 4
    assert 16 <= np.count_nonzero(sprite[..., 3]) < sprite[..., 3].size


def test_provisional_enemy_can_be_revalidated_after_restart(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner) + 2
    collision = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])
    assert learner.observe(collision, hit=True).learned_enemy_ids == ()

    restarted = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(restarted) + 2
    collision = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])
    confirmed = restarted.observe(collision, hit=True)

    assert confirmed.learned_enemy_ids
    assert not list((tmp_path / "learned_enemies" / "Game" / ".candidates").glob("*.png"))


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

    assert learner.observe(collision, hit=True).learned_enemy_ids == ()
    learner.observe(before, hit=False)
    revalidated_collision = before.copy()
    revalidated_collision[53, enemy_x + 5] = np.asarray([250, 60, 20], dtype=np.uint8)
    learned_ids = learner.observe(revalidated_collision, hit=True).learned_enemy_ids

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

    assert learner.observe(collision, hit=True).learned_enemy_ids == ()
    learner.observe(_frame(actor_x), hit=False)
    revalidated_collision = collision.copy()
    revalidated_collision[54, actor_x + 19] = np.asarray([245, 60, 20], dtype=np.uint8)
    learned_ids = learner.observe(revalidated_collision, hit=True).learned_enemy_ids

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
    assert learner.observe(collision, hit=True).learned_enemy_ids == ()
    learner.observe(_frame(actor_x), hit=False)
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


def test_version_two_enemy_is_demoted_until_a_new_contact_hit(tmp_path: Path, monkeypatch):
    root = tmp_path / "learned_enemies" / "Game"
    root.mkdir(parents=True)
    (root / ".learner-version").write_text("2", encoding="utf-8")
    sprite = np.zeros((14, 14, 4), dtype=np.uint8)
    sprite[1:13, 1:13, :3] = GOOMBA_COLOR[::-1]
    sprite[1:13, 1:13, 3] = 255
    assert cv2.imwrite(str(root / "old-enemy.png"), sprite)

    learner = _configure(tmp_path, monkeypatch)
    learner.observe(_frame(20), hit=False)

    assert learner.templates == {}
    assert "old-enemy" in learner.candidate_templates
    assert not list(root.glob("*.png"))
    assert (root / ".candidates" / "old-enemy.png").is_file()
    assert (root / ".learner-version").read_text(encoding="utf-8") == "3"


def test_near_identical_hit_crop_is_not_learned_twice(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner) + 2
    first = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])
    assert learner.observe(first, hit=True).learned_enemy_ids == ()

    learner.observe(_frame(actor_x), hit=False)
    variant = first.copy()
    variant[52:56, actor_x + 17 : actor_x + 21] = np.asarray([188, 64, 24], dtype=np.uint8)
    confirmed = learner.observe(variant, hit=True)

    learner.observe(_frame(actor_x), hit=False)
    duplicate = learner.observe(first, hit=True)

    assert confirmed.learned_enemy_ids
    assert duplicate.learned_enemy_ids == ()
    assert len(list((tmp_path / "learned_enemies" / "Game").glob("*.png"))) == 1


def test_detection_requires_two_consistent_frames(tmp_path: Path, monkeypatch):
    learner = _configure(tmp_path, monkeypatch)
    actor_x = _prime_actor(learner) + 2
    collision = _frame(actor_x, [(48, 60, actor_x + 13, actor_x + 25, GOOMBA_COLOR)])
    assert learner.observe(collision, hit=True).learned_enemy_ids == ()
    learner.observe(_frame(actor_x), hit=False)
    assert learner.observe(collision, hit=True).learned_enemy_ids

    learner.reset()
    actor_x = _prime_actor(learner)
    enemy_x = 70
    enemy_frame = _frame(actor_x, [(48, 60, enemy_x, enemy_x + 12, GOOMBA_COLOR)])

    transient = learner.observe(enemy_frame, hit=False)
    disappeared = learner.observe(_frame(actor_x + 2), hit=False)
    first_stable = learner.observe(_frame(actor_x + 4, [(48, 60, enemy_x, enemy_x + 12, GOOMBA_COLOR)]), hit=False)
    stable = learner.observe(_frame(actor_x + 6, [(48, 60, enemy_x, enemy_x + 12, GOOMBA_COLOR)]), hit=False)

    assert transient.detections == ()
    assert disappeared.detections == ()
    assert first_stable.detections == ()
    assert len(stable.detections) == 1
