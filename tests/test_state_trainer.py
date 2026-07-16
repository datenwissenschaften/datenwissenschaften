from types import SimpleNamespace

from datenwissenschaften import state_trainer
from datenwissenschaften.state_trainer import SavestateScheduler, StateTrainer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_savestate_scheduler_rotates_after_win():
    clock = Clock()
    scheduler = SavestateScheduler(("Level1", "Level2"), interval_seconds=14_400, clock=clock)

    assert scheduler.rotation_reason(won=True) == "successful episode"
    assert scheduler.rotate() == "Level2"


def test_savestate_scheduler_rotates_after_four_hours():
    clock = Clock()
    scheduler = SavestateScheduler(("Level1", "Level2"), interval_seconds=14_400, clock=clock)

    clock.now = 14_399
    assert scheduler.rotation_reason(won=False) is None
    clock.now = 14_400
    assert scheduler.rotation_reason(won=False) == "14400 seconds"
    assert scheduler.rotate() == "Level2"


def test_savestate_scheduler_does_not_rotate_single_savestate():
    clock = Clock()
    scheduler = SavestateScheduler(("Level1",), interval_seconds=14_400, clock=clock)
    clock.now = 20_000

    assert scheduler.rotation_reason(won=True) is None


def test_publish_state_training_includes_unreached_and_active_states(monkeypatch):
    published = {}
    monkeypatch.setattr(
        state_trainer,
        "publish_metadata",
        lambda section, values, *, replace=False: published.update(
            {"section": section, "values": values, "replace": replace}
        ),
    )
    models = {
        "Find": SimpleNamespace(num_timesteps=100, n_steps=16, n_envs=2),
        "Eat": SimpleNamespace(num_timesteps=0, n_steps=16, n_envs=2),
    }
    rollouts = SimpleNamespace(transitions={"Find": [object(), object()], "Eat": []})

    StateTrainer._publish_state_training(
        models,
        rollouts,
        ["Find", "Find"],
        {"Find": 3, "Eat": 0},
        {"Find": 2, "Eat": 0},
        {"Find": 12.5, "Eat": None},
        SimpleNamespace(ui=SimpleNamespace(enabled=True)),
    )

    assert published["section"] == "state_training"
    assert published["replace"] is True
    assert published["values"]["Find"] == {
        "active_environments": 2,
        "collected_steps": 100,
        "rollout_steps": 2,
        "rollout_capacity": 32,
        "model_updates": 2,
        "completed_segments": 3,
        "best_fitness": 12.5,
    }
    assert published["values"]["Eat"]["collected_steps"] == 0
