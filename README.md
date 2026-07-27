# datenwissenschaften library

A small state-driven reinforcement-learning engine for `stable-retro`.

The engine:

1. combines normalized RAM, active state, and template detection features;
2. lets game code define state transitions and reward;
3. trains one compact DQN policy across all states;
4. saves the policy after every training chunk;
5. saves state transitions and resumes from the first incomplete curriculum state.

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

Game states subclass `datenwissenschaften.states.state.State` and implement reward, termination, and transition
methods. Stable Baselines3 receives four stacked numeric feature observations and learns through a small DQN MLP.
A successful state transition atomically saves the emulator state for the next curriculum stage. After all stages are
complete, training mixes full runs with intermediate-state practice.

## State classes

| Builder type | Python base | Automatic behavior |
| --- | --- | --- |
| Default | `datenwissenschaften.states.state.State` | Updates frame and RAM, runs explicit scoring and outcomes |
| Image Detector | `datenwissenschaften.states.image_detector.ImageDetector` | Detects an exact uploaded template without adding reward |
| Target | `datenwissenschaften.states.target.TargetState` | Rewards distance progress and persists the first target location |
| Explorer | `datenwissenschaften.states.explorer.Explorer` | Rewards new positions and advances when the template is visible |

`ImageDetector`, `TargetState`, and `Explorer` require `template_file`. Explorer additionally requires
`position_x`, `position_y`, `screen_x`, and `screen_y` RAM fields. Target locations are stored under the game's model
directory and provide progress scoring after a target leaves the frame.

## Development

```bash
black --check src
ruff check src
ruff format --check src
python -m compileall -q src
```

## License

Distributed under the [GNU General Public License v3.0](LICENSE).
