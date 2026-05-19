from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Literal

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

MessageLevel = Literal["info", "ok", "warn"]


@dataclass
class ConsoleDashboard:
    console: Console = field(default_factory=lambda: Console(force_terminal=True))
    live: Live | None = None
    events: deque[tuple[str, MessageLevel, str]] = field(default_factory=lambda: deque(maxlen=7))
    game: str = "-"
    num_envs: int = 0
    best_time: int | None = None
    episodes: int = 0
    wins: int = 0
    steps: int = 0
    total_steps: int = 0
    last_env: int | None = None
    last_result: str = "-"
    last_save: str = "-"
    status: str = "Idle"
    _tried_start: bool = False

    def start(
        self,
        *,
        game: str,
        num_envs: int,
        best_time: int | None,
        total_steps: int = 0,
        current_steps: int = 0,
        announce: bool = True,
    ) -> None:
        self._tried_start = True
        self._require_real_terminal()

        self.game = game
        self.num_envs = num_envs
        self.best_time = best_time
        self.total_steps = total_steps
        self.steps = current_steps
        self.status = "Training"
        self.started_at = monotonic()

        if self.live is None:
            self.live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=2,
                transient=False,
                auto_refresh=True,
            )
            self.live.start()
        if announce:
            self.message("ok", f"Training started for {game} across {num_envs} envs")
        self.refresh()

    def stop(self) -> None:
        self.status = "Stopped"
        self.refresh()
        if self.live is not None:
            self.live.stop()
            self.live = None

    def message(self, level: MessageLevel, message: str) -> None:
        if self.live is None and not self._tried_start:
            self._start_from_environment()

        self._add_event(level, message)
        self.status = message
        if self.live is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.print(self._format_message(timestamp, level, message))
            return
        self.refresh()

    def record_episode(
        self,
        *,
        env_index: int,
        episode_index: int,
        won: bool,
        time_until_won: int,
        current_steps: int = 0,
    ) -> None:
        self.episodes += 1
        if current_steps:
            self.steps = current_steps
        if won:
            self.wins += 1
            self.last_result = f"win in {time_until_won}"
        else:
            self.last_result = "loss"
        self.last_env = env_index
        self.status = f"Env {env_index} finished episode {episode_index}: {self.last_result}"
        self.refresh()

    def record_best(self, *, time_until_won: int, bk2_path: str) -> None:
        self.best_time = time_until_won
        self.message("ok", f"New best: {time_until_won} frames ({Path(bk2_path).name})")

    def record_save(self, *, steps: int, model_path: str) -> None:
        self.steps = steps
        self.last_save = f"{steps:,} steps"
        self.status = f"Saved {Path(model_path).name}"
        self._add_event("info", f"Checkpoint saved at {steps:,} steps")
        self.refresh()

    def refresh(self) -> None:
        if self.live is not None:
            self.live.update(self._render(), refresh=True)

    def _render(self) -> Panel:
        return Panel(
            Group(
                self._header(),
                self._metrics(),
                self._progress(),
                self._activity(),
            ),
            title="[bold cyan]RETRO SPEEDLAB COMMAND CENTER[/]",
            border_style="cyan",
            box=box.DOUBLE,
        )

    def _header(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            self._kv("Game", self.game),
            self._kv("Status", self.status),
        )
        table.add_row(
            self._kv("Envs", str(self.num_envs)),
            self._kv("Checkpoint", self.last_save),
        )
        table.add_row(
            self._kv("Last Episode", self._last_episode()),
            self._kv("Best", str(self.best_time) if self.best_time is not None else "-"),
        )
        return table

    def _metrics(self) -> Table:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True, header_style="bold white")
        table.add_column("Steps", justify="right")
        table.add_column("Episodes", justify="right")
        table.add_column("Wins", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Best", justify="right")
        table.add_column("Ep/Min", justify="right")
        table.add_column("Uptime", justify="right")

        win_rate = f"{(self.wins / self.episodes * 100):.1f}%" if self.episodes else "0.0%"
        best = str(self.best_time) if self.best_time is not None else "-"
        rate = f"{self._episodes_per_minute():.2f}/min"
        table.add_row(
            f"{self.steps:,}",
            f"{self.episodes:,}",
            f"{self.wins:,}",
            win_rate,
            best,
            rate,
            self._uptime(),
        )
        return table

    def _progress(self) -> Progress:
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]Training Progress[/]"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>5.1f}%"),
            TextColumn("{task.completed:,.0f}/{task.total:,.0f} steps"),
            expand=True,
        )
        total = max(self.total_steps, self.steps, 1)
        progress.add_task("steps", total=total, completed=min(self.steps, total))
        return progress

    def _activity(self) -> Table:
        table = Table(box=box.SIMPLE, expand=True, show_header=True, header_style="bold white")
        table.add_column("Time", style="dim", width=8)
        table.add_column("Level", width=6)
        table.add_column("Event")

        if not self.events:
            table.add_row("-", "-", "Awaiting training events")
            return table

        for timestamp, level, message in self.events:
            table.add_row(timestamp, self._level_text(level), message)
        return table

    def _kv(self, label: str, value: str) -> Text:
        text = Text()
        text.append(f"{label}: ", style="bold white")
        text.append(str(value), style="cyan")
        return text

    def _format_message(self, timestamp: str, level: MessageLevel, message: str) -> Text:
        text = Text()
        text.append(f"{timestamp} ", style="dim")
        text.append(self._level_text(level))
        text.append(f" {message}")
        return text

    def _level_text(self, level: MessageLevel) -> Text:
        styles = {
            "info": "blue",
            "ok": "green",
            "warn": "yellow",
        }
        return Text(level.upper(), style=f"bold {styles[level]}")

    def _add_event(self, level: MessageLevel, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.appendleft((timestamp, level, message))

    def _last_episode(self) -> str:
        if self.last_env is None:
            return "-"
        return f"env {self.last_env}: {self.last_result}"

    def _elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(monotonic() - self.started_at, 0.0)

    def _episodes_per_minute(self) -> float:
        elapsed = self._elapsed_seconds()
        if elapsed == 0:
            return 0.0
        return self.episodes / elapsed * 60

    def _uptime(self) -> str:
        elapsed = int(self._elapsed_seconds())
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _start_from_environment(self) -> None:
        game = os.environ.get("RETRO_SPEEDLAB_GAME_ID")
        if not game:
            return

        self.start(
            game=game,
            num_envs=0,
            best_time=self._best_time_from_environment(game),
            current_steps=self._current_steps_from_environment(),
            announce=False,
        )

    def _best_time_from_environment(self, game: str) -> int | None:
        models_dir = os.environ.get("RETRO_SPEEDLAB_MODEL_DIR")
        if not models_dir:
            return None

        savestate = os.environ.get("RETRO_SPEEDLAB_SAVESTATE", "")
        best_time_path = Path(models_dir) / game / savestate / "best_time.txt"
        try:
            text = best_time_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not text.isdecimal():
            return None
        return int(text)

    def _current_steps_from_environment(self) -> int:
        value = os.environ.get("RETRO_SPEEDLAB_CURRENT_TIMESTEPS")
        if value and value.isdecimal():
            return int(value)
        return 0

    def _require_real_terminal(self) -> None:
        if os.environ.get("DEBUG") == "1":
            return
        if not (sys.stdout.isatty() and os.environ.get("TERM") != "dumb"):
            raise RuntimeError(
                "Retro Speedlab command center requires a real interactive terminal. "
                "PyCharm's Run console does not support the live dashboard; run this via the runner script."
            )


_dashboard = ConsoleDashboard()


def ui_start_training(
    *,
    game: str,
    num_envs: int,
    best_time: int | None,
    total_steps: int = 0,
    current_steps: int = 0,
) -> None:
    _dashboard.start(
        game=game,
        num_envs=num_envs,
        best_time=best_time,
        total_steps=total_steps,
        current_steps=current_steps,
    )


def ui_stop_training() -> None:
    _dashboard.stop()


def ui_info(message: str) -> None:
    _dashboard.message("info", message)


def ui_success(message: str) -> None:
    _dashboard.message("ok", message)


def ui_warning(message: str) -> None:
    _dashboard.message("warn", message)


def ui_episode_finished(
    *,
    env_index: int,
    episode_index: int,
    won: bool,
    time_until_won: int,
    current_steps: int = 0,
) -> None:
    _dashboard.record_episode(
        env_index=env_index,
        episode_index=episode_index,
        won=won,
        time_until_won=time_until_won,
        current_steps=current_steps,
    )


def ui_best_episode(*, time_until_won: int, bk2_path: str) -> None:
    _dashboard.record_best(time_until_won=time_until_won, bk2_path=bk2_path)


def ui_model_saved(*, steps: int, model_path: str) -> None:
    _dashboard.record_save(steps=steps, model_path=model_path)
