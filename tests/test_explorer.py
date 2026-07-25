from datenwissenschaften.helpers.position import Position
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.explorer import Coverage, Exploration, ExplorationSettings, Explorer
from datenwissenschaften.states.state import State


class _ConcreteExplorer(Explorer[RamInfo]):
    template_file = "unused.png"

    def _target_state(self) -> type[State[RamInfo]]:
        return State


def _explorer(x: int, y: int) -> _ConcreteExplorer:
    explorer = object.__new__(_ConcreteExplorer)
    position = Position(x, y)
    settings = ExplorationSettings.create(position.screen_size, explorer.step_penalty)
    explorer.exploration = Exploration.start(position, settings)
    return explorer


def test_frontier_rewards_sustained_horizontal_progress_but_not_backtracking():
    explorer = _explorer(100, 50)
    frontier = explorer.exploration.frontier

    first_progress = frontier.expand((104, 50), explorer.exploration.settings)
    backtrack = frontier.expand((102, 50), explorer.exploration.settings)
    further_progress = frontier.expand((112, 50), explorer.exploration.settings)

    assert first_progress == 4 / explorer.exploration.settings.screen_size
    assert backtrack == 0.0
    assert further_progress == 8 / explorer.exploration.settings.screen_size


def test_position_novelty_ignores_subtile_jitter():
    explorer = _explorer(100, 50)

    grid_size = explorer.exploration.settings.position_grid_size
    assert Coverage.bucket((100, 50), grid_size) == Coverage.bucket((103, 55), grid_size)
    assert Coverage.bucket((100, 50), grid_size) != Coverage.bucket((116, 50), grid_size)


def test_new_position_inside_existing_frontier_scores_and_resets_staleness():
    explorer = _explorer(100, 50)
    explorer.exploration.frontier.minimum_x = 80
    explorer.exploration.frontier.maximum_x = 140
    explorer.exploration.frontier.stale_steps = 30

    reward = explorer.exploration.reward((0, 0), (116, 50))

    assert reward == explorer.exploration.settings.position_reward
    assert explorer.exploration.frontier.stale_steps == 0


def test_coarse_region_discovery_adds_a_meaningful_coverage_bonus():
    explorer = _explorer(100, 50)
    explorer.exploration.frontier.minimum_x = 0
    explorer.exploration.frontier.maximum_x = 200

    reward = explorer.exploration.reward((0, 0), (132, 50))

    settings = explorer.exploration.settings
    assert reward == settings.region_reward + settings.position_reward


def test_revisit_penalty_adapts_to_staleness_and_is_bounded():
    explorer = _explorer(100, 50)
    frontier = explorer.exploration.frontier
    settings = explorer.exploration.settings
    frontier.stale_steps = settings.revisit_penalty_grace_steps
    assert frontier.revisit_penalty(100, settings) == 0.0

    frontier.stale_steps = settings.staleness_limit // 2
    mid_attempt = frontier.revisit_penalty(100, settings)
    frontier.stale_steps = settings.staleness_limit
    timed_out = frontier.revisit_penalty(100, settings)

    assert 0.0 < mid_attempt < timed_out
    assert timed_out < frontier.revisit_penalty(10_000, settings)


def test_revisit_pressure_grows_with_attempt_staleness():
    explorer = _explorer(100, 50)
    rewards = [explorer.exploration.reward((0, 0), (100, 50)) for _ in range(300)]

    assert rewards[0] == 0.0
    assert rewards[-1] < rewards[len(rewards) // 2] < 0.0


def test_implausible_coordinate_jump_does_not_poison_frontier_or_coverage():
    explorer = _explorer(100, 50)

    reward = explorer.exploration.reward((8, 8), (2200, 2200))

    assert reward <= 0.0
    assert explorer.exploration.frontier.maximum_x == 100
    assert explorer.exploration.frontier.maximum_y == 50
    assert (8, 8) not in explorer.exploration.coverage.areas
    assert explorer.exploration.invalid_position_steps == 1


def test_frontier_reward_adapts_equally_to_each_game_direction():
    horizontal_explorer = _explorer(100, 50)
    vertical_explorer = _explorer(100, 50)
    horizontal = horizontal_explorer.exploration.frontier.expand((110, 50), horizontal_explorer.exploration.settings)
    vertical = vertical_explorer.exploration.frontier.expand((100, 60), vertical_explorer.exploration.settings)

    assert horizontal == vertical


def test_offscreen_goal_is_not_penalized_during_exploration():
    assert _ConcreteExplorer.target_missing_penalty == 0.0


def test_target_speed_reward_is_higher_when_target_is_reached_faster():
    explorer = _explorer(100, 50)

    fast_reward = explorer._target_speed_reward(100.0, 100)
    slow_reward = explorer._target_speed_reward(100.0, 500)

    assert fast_reward > slow_reward > 0.0


def test_target_speed_reward_scales_with_exploration_required_by_game():
    explorer = _explorer(100, 50)

    reward = explorer._target_speed_reward(2400.0, 16)

    assert reward == 600.0


def test_explorer_truncates_after_significant_frontier_staleness():
    explorer = _explorer(100, 50)
    explorer.target_detector = type("Detector", (), {"seen": False})()
    limit = explorer.exploration.settings.staleness_limit

    explorer.exploration.frontier.stale_steps = limit - 1
    assert explorer._truncated() is False

    explorer.exploration.frontier.expand((100, 50), explorer.exploration.settings)
    assert explorer._truncated() is True


def test_frontier_progress_resets_staleness_timeout():
    explorer = _explorer(100, 50)
    explorer.target_detector = type("Detector", (), {"seen": False})()
    explorer.exploration.frontier.stale_steps = explorer.exploration.settings.staleness_limit - 1

    explorer.exploration.frontier.expand((101, 50), explorer.exploration.settings)

    assert explorer.exploration.frontier.stale_steps == 0
    assert explorer._truncated() is False


def test_detected_target_takes_precedence_over_staleness_timeout():
    explorer = _explorer(100, 50)
    explorer.target_detector = type("Detector", (), {"seen": True})()
    explorer.exploration.frontier.stale_steps = explorer.exploration.settings.staleness_limit

    assert explorer._truncated() is False


def test_staleness_feature_tracks_the_full_timeout():
    explorer = _explorer(100, 50)
    limit = explorer.exploration.settings.staleness_limit
    explorer.exploration.frontier.stale_steps = limit // 2

    features = explorer.exploration.frontier.features(Position(100, 50), explorer.exploration.settings)

    assert features[-1] == 0.5


def test_exploration_settings_adapt_to_game_coordinate_scale():
    small = ExplorationSettings.create(144, 0.05)
    large = ExplorationSettings.create(400, 0.05)

    assert small.position_grid_size < large.position_grid_size
    assert small.region_grid_size < large.region_grid_size
    assert small.staleness_limit < large.staleness_limit
