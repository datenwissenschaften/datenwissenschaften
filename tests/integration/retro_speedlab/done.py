from datenwissenschaften.states.image_detector import ImageDetector

from tests.integration.retro_speedlab.ram import AirstrikerRam


class Done(ImageDetector[AirstrikerRam]):
    template_file: str

    def _won(self) -> bool:
        return self.target_detector is not None and self.target_detector.seen
