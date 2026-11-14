# Dr. Martin Franke

**Senior AI Engineer · Machine Learning Engineer · Lead Data Scientist**

Dr.-Ing. in Computer Science with 15+ years of experience across applied machine learning, AI, experimentation, and large-scale data systems.

I build production-oriented ML and analytics systems from research and prototyping through architecture, implementation, validation, deployment, and governance. My current technical focus is **agentic AI systems (Claude agents built with Claude Code and the Agent SDK, with evaluation and observability), search and reinforcement learning, and production data and ML systems**.

## Selected projects

| Project | Focus |
|---|---|
| [retro-speedlab-core](https://github.com/datenwissenschaften/retro-speedlab-core) | Engine of [Retro Speedlab](https://www.retrospeedlab.com), an AI that speedruns its own AI-designed 8-bit games for Twitch: Go-Explore, beam and tree search over RAM save states guided by a Stable-Baselines3 PPO policy, every run verified frame-exact from power-on, Claude Code lab runs traced in Langfuse and scored on search progress, experiments in Weights & Biases |
| retro-speedlab-maintainer (private) | Scheduled Claude Agent SDK agent that keeps Retro Speedlab running: every six hours it reads Bugsink issues, CI, lab reports and stream engagement, fixes the most important problem with a test, then merges and deploys, or opens an issue when it cannot |
| retro-speedlab-engine (private) | The console the games run on: a Pyxel cartridge contract with all game state in RAM, a headless runner for the search, movies, videos with sound and browser builds |
| retro-speedlab-games (private) | [LAB50](https://games.retrospeedlab.com), a growing collection of original 8-bit games designed by Claude agents, each with one twist, playable in the browser and built to be speedrun by the search |
| [agentic-data-analyst](https://github.com/datenwissenschaften/agentic-data-analyst) | Chat with a speedrun database: Claude writes read-only DuckDB SQL, a reviewer agent checks it, and a Chainlit UI shows answers with tables and charts; deterministic SQL guardrails and cost budgets |
| [spark-data-quality](https://github.com/datenwissenschaften/spark-data-quality) | Typed, Spark-native data-quality validation and profiling with shared aggregation planning |
| [experimentation-toolkit](https://github.com/datenwissenschaften/experimentation-toolkit) | A/B testing, statistical inference, SRM diagnostics, power analysis, and multiple-testing correction |
| [spark-asammdf](https://github.com/datenwissenschaften/spark-asammdf) | Apache Spark DataSource V2 connector for distributed analysis of ASAM MDF4 measurement data |

## Engineering focus

**AI agents & LLMs:** Claude Code · Claude Agent SDK · MCP · multi-agent orchestration · LLM evaluation · Langfuse · OpenTelemetry

**ML & search:** PyTorch · Stable-Baselines3 · Gymnasium · reinforcement learning · search-based planning · scikit-learn · Spark ML · Weights & Biases

**Data & distributed systems:** PySpark · Apache Spark · Databricks · Airflow · Flyte · SQL · DuckDB · PostgreSQL · data quality · metadata · lineage

**Software engineering:** Python · pyright · ruff · Scala · FastAPI · Vue · REST · Docker · Dokku · GitHub Actions · CI/CD · testing · reproducibility

**Cloud:** Azure · AWS · GCP · Hetzner

## Background

I started working on applied machine learning and behavioral analytics at **TU Dresden**, where I completed my doctorate in Computer Science.

Since then, I have worked across research, healthcare, digital products, and automotive. My recent work has centered on production ML and large-scale automotive analytics: distributed PySpark pipelines, semantic data abstractions and metadata indexing, ML/AI-assisted analytics, reproducible delivery, monitoring, and governance.

The repositories above use public or synthetic data and are independent portfolio projects; they do not contain proprietary employer or client code or data.

---

[Website](https://www.datenwissenschaften.com/?utm_source=github&utm_medium=profile&utm_campaign=portfolio&utm_content=profile-readme)
