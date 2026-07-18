from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.explorer import Explorer
from datenwissenschaften.states.state import State


class _ConcreteExplorer(Explorer[RamInfo]):
    template_file = "unused.png"

    def _target_state(self) -> type[State[RamInfo]]:
        return State


def _explorer(x: int = 100, y: int = 50) -> _ConcreteExplorer:
    explorer = object.__new__(_ConcreteExplorer)
    explorer.frontier_min_x = explorer.frontier_max_x = x
    explorer.frontier_min_y = explorer.frontier_max_y = y
    explorer.steps_since_frontier = 0
    return explorer


def test_frontier_rewards_sustained_horizontal_progress_but_not_backtracking():
    explorer = _explorer()

    first_progress = explorer._frontier_reward((104, 50))
    backtrack = explorer._frontier_reward((102, 50))
    further_progress = explorer._frontier_reward((112, 50))

    assert first_progress == 4.0
    assert backtrack == 0.0
    assert further_progress == 8.0


def test_position_novelty_ignores_subtile_jitter():
    explorer = _explorer()

    assert explorer._position_bucket((100, 50)) == explorer._position_bucket((103, 55))
    assert explorer._position_bucket((100, 50)) != explorer._position_bucket((108, 50))


def test_frontier_stall_penalty_waits_for_grace_period_and_is_bounded():
    explorer = _explorer()

    grace_rewards = [explorer._frontier_reward((100, 50)) for _ in range(explorer.frontier_stall_grace_steps)]
    first_penalty = explorer._frontier_reward((100, 50))
    for _ in range(100):
        final_penalty = explorer._frontier_reward((100, 50))

    assert grace_rewards == [0.0] * explorer.frontier_stall_grace_steps
    assert first_penalty == -explorer.frontier_stall_penalty_scale
    assert final_penalty == -explorer.maximum_frontier_stall_penalty


def test_vertical_frontier_has_smaller_reward_than_horizontal_frontier():
    horizontal = _explorer()._frontier_reward((110, 50))
    vertical = _explorer()._frontier_reward((100, 60))

    assert horizontal == 10.0
    assert vertical == 1.0


def test_offscreen_goal_is_not_penalized_during_exploration():
    assert _ConcreteExplorer.target_missing_penalty == 0.0


def test_explorer_truncates_after_significant_frontier_staleness():
    explorer = _explorer()
    explorer.target_detector = type("Detector", (), {"seen": False})()

    explorer.steps_since_frontier = explorer.frontier_staleness_limit - 1
    assert explorer._truncated() is False

    explorer._frontier_reward((100, 50))
    assert explorer._truncated() is True


def test_frontier_progress_resets_staleness_timeout():
    explorer = _explorer()
    explorer.target_detector = type("Detector", (), {"seen": False})()
    explorer.steps_since_frontier = explorer.frontier_staleness_limit - 1

    explorer._frontier_reward((101, 50))

    assert explorer.steps_since_frontier == 0
    assert explorer._truncated() is False


def test_detected_target_takes_precedence_over_staleness_timeout():
    explorer = _explorer()
    explorer.target_detector = type("Detector", (), {"seen": True})()
    explorer.steps_since_frontier = explorer.frontier_staleness_limit

    assert explorer._truncated() is False
