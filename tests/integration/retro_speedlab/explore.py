from datenwissenschaften.states.explorer import Explorer
from datenwissenschaften.states.state import State

from tests.integration.retro_speedlab.ram import AirstrikerRam
from tests.integration.retro_speedlab.target import ApproachTarget


class ExploreTarget(Explorer[AirstrikerRam]):
    template_file: str

    def _target_state(self) -> type[State[AirstrikerRam]]:
        return ApproachTarget
