# datenwissenschaften: Retro Speedlab Core 🚀

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**datenwissenschaften** is the core engine powering [Retro Speedlab](https://github.com/datenwissenschaften/retro-speedlab), a high-performance Reinforcement Learning (RL) toolkit for classic video games. Built on top of `stable-baselines3` and `stable-retro`, it provides the underlying infrastructure for training, monitoring, and recording RL agents.

## ⚠️ Important Note

This package is intended as the internal library for the Retro Speedlab project. For the full experience—including automated runners, training scripts, and comprehensive documentation—please use the main repository:

👉 **[https://github.com/datenwissenschaften/retro-speedlab](https://github.com/datenwissenschaften/retro-speedlab)**

## ✨ Features

*   **🎮 Command Center**: A rich terminal dashboard for real-time training metrics.
*   **🏋️ Orchestrated Training**: Simplified RL workflows and session management.
*   **⚡ GPU acceleration**: CUDA tuning for PPO plus batched GPU inference for NEAT, with CPU fallback.
*   **📊 Smart Callbacks**: Automatic checkpointing and replay recording (`.bk2`).
*   **🛠️ Robust Infrastructure**: Streamlined environment and ROM management.

## 🚀 Installation

```bash
pip install datenwissenschaften
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust its values. Application settings are read from YAML rather than
environment variables. APIs that load configuration also accept an explicit `config_path` when the file is stored
elsewhere. Relative paths in the file are resolved relative to the configuration file.

Set `training.num_envs` to `auto` to select parallel environment workers from CPU affinity, population size, and the
systemd/cgroup memory limit. An explicit positive integer continues to override automatic selection.

Set `ui.enable: true` to run the local Vue training dashboard at `http://127.0.0.1:18080`. It charts episode fitness
and step counts, shows termination outcomes, and reports environment, PPO, and NEAT configuration details. The `ui`
mapping accepts `enable`, `host`, `port`, and `max_episodes` values. Dashboard history
is restored from and atomically persisted to `models/<game>/<savestate>/history.json` (relative to the configured
models directory). Set `host: 0.0.0.0` to listen on all network interfaces; use the machine's IP address rather than
`0.0.0.0` when opening the dashboard from another device.

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---
Developed with ❤️ by [datenwissenschaften](https://www.datenwissenschaften.com).
