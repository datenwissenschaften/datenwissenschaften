# datenwissenschaften library

A small state-driven reinforcement-learning engine for `stable-retro`.

The engine:

1. combines a compact grayscale scene with RAM, state, target, motion, and prior-action features;
2. provides automatic exploration and target-progress rewards;
3. trains one compact recurrent PPO policy across all states;
4. saves the policy throughout training.

## Installation

```bash
git clone https://git.datenwissenschaften.com/datenwissenschaften/datenwissenschaften.git
cd datenwissenschaften
poetry install
```

Python 3.12 is required.

## Training

```python
from pathlib import Path

from datenwissenschaften.retro.environment import build_environment
from datenwissenschaften.training.trainer import train

from game.wrapper import GameWrapper

config_path = Path("config.yaml")
environment = build_environment(GameWrapper, config_path)
train(environment, config_path)
```

Game states subclass `Explorer` or `TargetState` and implement outcomes and transitions. Stable Baselines3 receives
a visual and numeric observation and learns through a compact CNN with a shared LSTM policy.

## State classes

| Builder type | Python base | Automatic behavior |
| --- | --- | --- |
| Target | `datenwissenschaften.states.target.TargetState` | Rewards distance progress and persists the first target location |
| Explorer | `datenwissenschaften.states.explorer.Explorer` | Rewards new positions and advances when the template is visible |

Every project requires `screen_x`, `screen_y`, `player_x`, and `player_y` RAM fields. Player velocity is calculated
from consecutive positions. Every state requires `template_file`. Target locations are stored under the game's model
directory and provide progress scoring after a target leaves the frame. A target state defines exactly one of
`_next()` or `_won()`.

## Development

```bash
black --check src
ruff check src
ruff format --check src
python -m compileall -q src
```

## License

Distributed under the [GNU General Public License v3.0](LICENSE).
