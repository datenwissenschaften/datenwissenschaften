from types import SimpleNamespace

from datenwissenschaften import state_trainer
from datenwissenschaften.state_trainer import StateTrainer


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
        1_000,
        SimpleNamespace(ui=SimpleNamespace(enabled=True)),
    )

    assert published["section"] == "state_training"
    assert published["replace"] is True
    assert published["values"]["Find"] == {
        "active_environments": 2,
        "collected_steps": 100,
        "target_steps": 1_000,
        "progress_percent": 10.0,
        "rollout_steps": 2,
        "rollout_capacity": 32,
        "model_updates": 2,
        "completed_segments": 3,
    }
    assert published["values"]["Eat"]["collected_steps"] == 0
