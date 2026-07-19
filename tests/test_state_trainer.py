from datetime import datetime
from types import SimpleNamespace

from datenwissenschaften import state_trainer
from datenwissenschaften.state_trainer import SavestateScheduler, StateTrainer


class Clock:
    def __init__(self):
        self.now = datetime(2026, 7, 17, 23, 0)

    def __call__(self):
        return self.now


def test_savestate_scheduler_does_not_rotate_after_win():
    clock = Clock()
    scheduler = SavestateScheduler(("Level1", "Level2"), clock=clock)

    assert scheduler.rotation_reason(won=True) is None


def test_savestate_scheduler_rotates_once_after_local_midnight():
    clock = Clock()
    scheduler = SavestateScheduler(("Level1", "Level2"), clock=clock)

    clock.now = datetime(2026, 7, 17, 23, 59, 59)
    assert scheduler.rotation_reason(won=False) is None
    clock.now = datetime(2026, 7, 18, 0, 0)
    assert scheduler.rotation_reason(won=False) == "local midnight"
    assert scheduler.rotate() == "Level2"
    clock.now = datetime(2026, 7, 18, 12, 0)
    assert scheduler.rotation_reason(won=True) is None


def test_savestate_scheduler_does_not_rotate_single_savestate():
    clock = Clock()
    scheduler = SavestateScheduler(("Level1",), clock=clock)
    clock.now = datetime(2026, 7, 18, 12, 0)

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


def test_publish_curriculum_progress_uses_environment_values(monkeypatch):
    published = {}
    progress = {"Eat": {"wins": 3, "win_target": 8, "bad_checkpoint_evidence_target": 32}}
    venv = SimpleNamespace(env_method=lambda method: [progress])
    monkeypatch.setattr(
        state_trainer,
        "publish_metadata",
        lambda section, values, *, replace=False: published.update(
            {"section": section, "values": values, "replace": replace}
        ),
    )

    StateTrainer._publish_curriculum_progress(venv, SimpleNamespace(ui=SimpleNamespace(enabled=True)))

    assert published == {"section": "savestate_curriculum", "values": progress, "replace": True}


def test_state_model_update_keeps_configured_minibatch_size(monkeypatch, tmp_path):
    class Model:
        batch_size = 256

        def __init__(self):
            self.trained_batch_size = None

        def train(self):
            self.trained_batch_size = self.batch_size

    model = Model()
    rollouts = SimpleNamespace(build_buffer=lambda state_name: SimpleNamespace(buffer_size=8192))
    trainer = object.__new__(StateTrainer)
    trainer.config_path = "unused.yaml"
    monkeypatch.setattr(
        state_trainer,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="game"),
        ),
    )
    monkeypatch.setattr(state_trainer, "atomic_save", lambda model, path: None)

    trainer._update_model("ActivateScale", model, rollouts)

    assert model.rollout_buffer.buffer_size == 8192
    assert model.trained_batch_size == 256
    assert model.batch_size == 256
