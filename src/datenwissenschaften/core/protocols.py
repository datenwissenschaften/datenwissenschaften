from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from stable_baselines3.common.callbacks import BaseCallback


class TrainableModel(Protocol):
    num_timesteps: int

    def learn(
        self,
        total_timesteps: int,
        callback: Sequence[BaseCallback],
        reset_num_timesteps: bool,
    ) -> Any: ...


VenvBuilder = Callable[[int, int], Any]
ModelBuilder = Callable[[Any], TrainableModel]
Initializer = Callable[[str], None]
CallbackFactory = Callable[[], Sequence[BaseCallback]]
