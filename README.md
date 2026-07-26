# datenwissenschaften library

A small recurrent reinforcement-learning engine for `stable-retro`.

The engine has four responsibilities:

1. turn emulator frames, RAM, and the active game state into one Gymnasium observation;
2. let game code define state transitions and reward;
3. train one `sb3-contrib` recurrent PPO policy across all states;
4. save the policy after every rollout.

Each state must be beaten 16 times before its emulator savestate becomes the next curriculum starting point.
An automatic savestate is discarded after 128 attempts without a better episode score.
PPO owns its optimization parameters. The engine does not tune learning rates or use separate per-state models.

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
methods. A single recurrent policy receives the visual frame, normalized RAM, and a one-hot active-state signal.

## State classes

| Builder type | Python base | Automatic behavior |
| --- | --- | --- |
| Default | `datenwissenschaften.states.state.State` | Updates frame and RAM, runs explicit scoring and outcomes |
| Image Detector | `datenwissenschaften.states.image_detector.ImageDetector` | Detects an exact uploaded template without adding reward |
| Target | `datenwissenschaften.states.target.TargetState` | Adds proximity reward and persists the first target location |
| Explorer | `datenwissenschaften.states.explorer.Explorer` | Adds count-based position novelty and advances when the template is visible |

`ImageDetector`, `TargetState`, and `Explorer` require `template_file`. Explorer additionally requires
`position_x`, `position_y`, `screen_x`, and `screen_y` RAM fields. Target locations are stored under the game's model
directory and provide proximity scoring after a target leaves the frame.

## Development

```bash
black --check src
ruff check src
ruff format --check src
python -m compileall -q src
```

## License

Distributed under the [GNU General Public License v3.0](LICENSE).
