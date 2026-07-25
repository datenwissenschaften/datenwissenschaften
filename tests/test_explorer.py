from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.explorer import Explorer
from datenwissenschaften.states.state import State


class _ConcreteExplorer(Explorer[RamInfo]):
    template_file = "unused.png"

    def _target_state(self) -> type[State[RamInfo]]:
        return State


def _explorer(x: int = 100, y: int = 50) -> _ConcreteExplorer:
    explorer = object.__new__(_ConcreteExplorer)
    position_bucket = explorer._position_bucket((x, y))
    explorer.visited_areas = {(0, 0)}
    explorer.visited_regions = {explorer._region_bucket((x, y))}
    explorer.visited_positions = {position_bucket}
    explorer.position_visit_counts = {position_bucket: 1}
    explorer.frontier_min_x = explorer.frontier_max_x = x
    explorer.frontier_min_y = explorer.frontier_max_y = y
    explorer.steps_since_frontier = 0
    explorer.previous_coordinates = (x, y)
    explorer.invalid_position_steps = 0
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


def test_new_position_inside_existing_frontier_scores_and_resets_staleness():
    explorer = _explorer()
    explorer.frontier_min_x = 80
    explorer.frontier_max_x = 140
    explorer.steps_since_frontier = 30

    reward = explorer._exploration_reward((0, 0), (108, 50))

    assert reward == explorer.position_discovery_reward
    assert explorer.steps_since_frontier == 0


def test_coarse_region_discovery_adds_a_meaningful_coverage_bonus():
    explorer = _explorer()
    explorer.frontier_min_x = 0
    explorer.frontier_max_x = 200

    reward = explorer._exploration_reward((0, 0), (132, 50))

    assert reward == explorer.region_discovery_reward + explorer.position_discovery_reward


def test_revisit_penalty_adapts_to_staleness_and_is_bounded():
    explorer = _explorer()
    explorer.steps_since_frontier = explorer.revisit_penalty_grace_steps
    assert explorer._adaptive_revisit_penalty(100) == 0.0

    explorer.steps_since_frontier = explorer.frontier_staleness_limit // 2
    mid_attempt = explorer._adaptive_revisit_penalty(100)
    explorer.steps_since_frontier = explorer.frontier_staleness_limit
    timed_out = explorer._adaptive_revisit_penalty(100)

    assert 0.0 < mid_attempt < timed_out
    assert timed_out <= explorer.maximum_revisit_penalty
    assert explorer._adaptive_revisit_penalty(10_000) == explorer.maximum_revisit_penalty


def test_revisits_never_accumulate_the_previous_large_failure_incentive():
    explorer = _explorer()
    rewards = [explorer._exploration_reward((0, 0), (100, 50)) for _ in range(600)]

    assert min(rewards) >= -explorer.maximum_revisit_penalty
    assert sum(rewards) > -100.0


def test_implausible_coordinate_jump_does_not_poison_frontier_or_coverage():
    explorer = _explorer()

    reward = explorer._exploration_reward((8, 8), (2200, 2200), screen_size=256)

    assert reward <= 0.0
    assert explorer.frontier_max_x == 100
    assert explorer.frontier_max_y == 50
    assert (8, 8) not in explorer.visited_areas
    assert explorer.invalid_position_steps == 1


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


def test_staleness_feature_tracks_the_full_timeout():
    explorer = _explorer()
    explorer.steps_since_frontier = explorer.frontier_staleness_limit // 2
    ram = RamInfo()
    object.__setattr__(ram, "position_x", 100)
    object.__setattr__(ram, "position_y", 50)

    features = explorer._exploration_features(ram)

    assert features[-1] == 0.5
