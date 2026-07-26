from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

from tests.integration.retro_speedlab.done import Done
from tests.integration.retro_speedlab.ram import AirstrikerRam


class ApproachTarget(TargetState[AirstrikerRam]):
    template_file: str

    def _next(self) -> type[State[AirstrikerRam]] | None:
        if self.target_detector is not None and self.target_detector.seen:
            return Done
        return None
