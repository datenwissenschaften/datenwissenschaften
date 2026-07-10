# datenwissenschaften

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

The reinforcement-learning engine behind [Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab).

`datenwissenschaften` turns classic-game emulators into reproducible training systems. It provides visual and
state-aware environments, recurrent PPO agents, parallel execution, durable checkpoints, episode
recording, and a live browser dashboard in one focused Python package.

> This repository contains the reusable engine. For game runners, end-to-end examples, and user-facing
> documentation, start with [Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab).

## Why this engine

- **Exploration for sparse rewards** — multi-input CNN-LSTM PPO combines visual frames, normalized RAM, temporal
  memory, and normalized, clipped, annealed Random Network Distillation (RND).
- **Efficient execution** — vectorized environments, automatic worker selection, CUDA tuning, and CPU fallback.
- **Reliable training runs** — atomic checkpoints, resumable model state, and `.bk2` replay capture.
- **Operational visibility** — a local Vue dashboard reports episode outcomes, reward distributions, environment
  details, PPO parameters, and RND progress.
- **Game-oriented infrastructure** — ROM discovery, RAM models, state machines, visual encoders, and configurable
  action translation.

## Model choices

| Model | Best suited to | Characteristics |
| --- | --- | --- |
| `RecurrentRNDModel` | Sparse-reward NES games and partially observable state | Automatic visual + RAM inputs, NES-tuned recurrent PPO, LSTM memory, intrinsic RND exploration |
| Custom SB3 model | Experiments that need a standard Stable-Baselines3 algorithm | Integrates through the same builder, trainer, callbacks, and dashboard |

`RecurrentRNDModel` is the recommended starting point for visual agents. RND encourages the policy to visit novel
observations, while its influence decays during training so learned external rewards increasingly drive behavior.
The predictor, fixed target, optimizer, reward statistics, and annealing progress are all preserved in checkpoints.
Environment wrappers always use RGB observations and one emulator step per selected action; these are fixed engine
defaults rather than game-level options.
The default profile uses longer 512-step rollouts, a 512-unit LSTM, `gamma=0.999`, `gae_lambda=0.98`, and a slower
10-million-step RND decay. These settings preserve more temporal context and delayed reward information than the
shorter arcade baseline while retaining conservative PPO updates.

## Installation

The package requires Python 3.12.

```bash
pip install datenwissenschaften
```

For local development:

```bash
git clone https://github.com/datenwissenschaften/datenwissenschaften.git
cd datenwissenschaften
poetry install
cp config.example.yaml config.yaml
```

## Training dashboard

Enable the dashboard with `ui.enable: true`, then open [http://127.0.0.1:18080](http://127.0.0.1:18080). It refreshes
live training telemetry without interrupting the learner.

Dashboard history is restored from and persisted to Redis. The default Redis URL is
`redis://127.0.0.1:6379/0`, and history keys use the `datenwissenschaften:history` prefix. The `ui` mapping accepts
`enable`, `host`, `port`, `max_episodes`, `redis_url`, and `history_key_prefix`. Snapshots retain the latest
1,000 episodes by default and include summarized totals for discarded episodes; set `max_episodes` to another positive
integer or `null` for unlimited retained rows.
Binding to `0.0.0.0` makes the dashboard reachable on the local network; use that only on a trusted network and open
it through the machine's actual IP address.

## How a run fits together

1. A game package defines RAM structures, training states, rewards, and action translation.
2. The environment factory creates vectorized emulator workers and processed visual observations.
3. A model builder creates or restores the selected policy.
4. The trainer coordinates learning, checkpoints, replay capture, telemetry, and optional uploads.
5. The dashboard exposes the active run without coupling the learner to a separate monitoring service.

## Development

Run Python quality checks:

```bash
ruff check src
black --check src
python -m compileall -q src
```

Build the dashboard assets after changing the Vue frontend:

```bash
cd src/datenwissenschaften/ui/frontend
npm ci
npm run build
```

## License

Copyright © datenwissenschaften contributors. Distributed under the [GNU General Public License v3.0](LICENSE).
