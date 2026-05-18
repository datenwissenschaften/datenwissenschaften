from __future__ import annotations

from collections.abc import Sequence

from datenwissenschaften.console import ui_info
from datenwissenschaften.core.config import TrainingConfig
from datenwissenschaften.core.protocols import (
    CallbackFactory,
    Initializer,
    ModelBuilder,
    TrainableModel,
    VenvBuilder,
)


class TrainingSession:
    def __init__(
        self,
        *,
        config: TrainingConfig,
        build_venv: VenvBuilder,
        build_model: ModelBuilder,
        initializers: Sequence[Initializer] | None = None,
        callbacks: CallbackFactory | None = None,
    ) -> None:
        self.config = config
        self._build_venv = build_venv
        self._build_model = build_model
        self._initializers = list(initializers or [])
        self._callbacks = callbacks or (lambda: [])
        self._active_game: str | None = None

    def initialize(self) -> None:
        self._claim_game(self.config.game)
        for initializer in self._initializers:
            initializer(self.config.game)

    def build(self) -> TrainableModel:
        self._claim_game(self.config.game)
        venv = self._build_venv(self.config.num_envs, self.config.n_stack)
        return self._build_model(venv)

    def train_forever(self, model: TrainableModel) -> None:
        self._claim_game(self.config.game)
        ui_info(f"Training {self.config.game} with {self.config.num_envs} envs")

        while True:
            model.learn(
                total_timesteps=model.num_timesteps + self.config.timestep_batch,
                callback=list(self._callbacks()),
                reset_num_timesteps=False,
            )

    def run(self) -> TrainableModel:
        self.initialize()
        model = self.build()
        self.train_forever(model)
        return model

    def _claim_game(self, game: str) -> None:
        if self._active_game is None:
            self._active_game = game
            return
        if self._active_game != game:
            raise RuntimeError(
                f"TrainingSession is already bound to {self._active_game!r}. "
                f"Create a new session to train {game!r}."
            )
