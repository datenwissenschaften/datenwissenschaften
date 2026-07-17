from datenwissenschaften.ui.telemetry import _coerce_summary, _empty_summary, _summarize_episode


def test_episode_history_is_grouped_by_savestate():
    summary = _empty_summary()

    _summarize_episode(
        summary,
        {
            "index": 1,
            "savestate": "Level1",
            "training_state": "Run",
            "fitness": 12.0,
            "won": True,
            "duration_seconds": 3.0,
        },
    )
    _summarize_episode(
        summary,
        {
            "index": 2,
            "savestate": "Level2",
            "training_state": "Run",
            "fitness": 4.0,
            "won": False,
            "duration_seconds": 1.0,
        },
    )

    assert summary["episodes"] == 2
    assert summary["by_savestate"]["Level1"]["episodes"] == 1
    assert summary["by_savestate"]["Level1"]["wins"] == 1
    assert summary["by_savestate"]["Level2"]["wins"] == 0


def test_persisted_savestate_history_is_loaded():
    coerced = _coerce_summary(
        {
            "episodes": 3,
            "by_savestate": {
                "Level1": {"episodes": 2, "wins": 1},
                "Level2": {"episodes": 1, "wins": 0},
            },
        }
    )

    assert coerced is not None
    assert coerced["by_savestate"]["Level1"]["episodes"] == 2
    assert coerced["by_savestate"]["Level2"]["episodes"] == 1
