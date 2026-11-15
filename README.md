# Dr. Martin Franke

**Senior AI Engineer · Machine Learning Engineer · Lead Data Scientist**

Dr.-Ing. in Computer Science with 15+ years of experience across applied machine learning, AI, experimentation, and large-scale data systems.

I build production-oriented ML and analytics systems from research and prototyping through architecture, implementation, validation, deployment, and governance. My current technical focus is **agentic AI systems (Claude agents built with Claude Code and the Agent SDK, with evaluation and observability), search and reinforcement learning, and production data and ML systems**.

## Selected projects

| Project | Focus |
|---|---|
| [retro-speedlab-core](https://github.com/datenwissenschaften/retro-speedlab-core) | TAS search of [Retro Speedlab](https://www.retrospeedlab.com), an AI that speedruns its own AI-written 8-bit games live on Twitch: Go-Explore, backtracking, beam and tree search over RAM save states with a Stable-Baselines3 PPO policy trained on the GPU, every new part counted only once the whole movie replays frame-exact from power-on and then searched again for fewer frames, runs tracked in Weights & Biases and errors in Bugsink |
| retro-speedlab-maintainer (private) | Scheduled Claude Agent SDK agent that keeps Retro Speedlab running: every six hours on black-omen it reads Bugsink issues, CI, the live run and stream engagement, fixes the most important problem with a test and merges it into main, from where the website redeploys, or opens a pull request; what it cannot fix becomes an issue |
| retro-speedlab-engine (private) | The deterministic console the games run on (2 KB game memory, 60 fps, 16 colours): a Pyxel cartridge contract with all game state in RAM and an explicit unlock flag at power-on, a headless runner for the search, movies, videos with sound and browser builds that remember a won campaign |
| retro-speedlab-games (private) | [LAB24](https://games.retrospeedlab.com), a collection of 24 original 8-bit games written by an AI, each with one twist, playable in the browser: a campaign built to be speedrun by the search, and an endless mode of seeded, solvability-tested generated levels that unlocks once the campaign is won |
| retro-speedlab-runner (private) | Runs the search on black-omen, a home desktop, as one Dokku app: three LAB24 games are searched at once in two-hour turns, each resuming its own progress, the game on stream also trains a PPO policy on the GPU, and every game's run is relayed to the website; a scheduled Claude Code lab job improves a stuck game's speedrun package and ships it only when its checks pass and the movie still replays |
| retro-speedlab-stream (private) | The [live stream overlay](https://stream.retrospeedlab.com): the search's newest run, splits and attempts as an OBS browser source for Twitch |
| [retro-speedlab-lakehouse](https://github.com/datenwissenschaften/retro-speedlab-lakehouse) | The data platform of Retro Speedlab: a relay from the backend's outbox to Kafka (KRaft), Spark Structured Streaming and Airflow-scheduled PySpark batch jobs on Delta Lake with MERGE, time travel, OPTIMIZE/ZORDER and VACUUM; bronze stays on black-omen, the silver, gold and live tables the website serves live in an S3 lake the backend reads with delta-rs, and every write passes a spark-data-quality gate |
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
