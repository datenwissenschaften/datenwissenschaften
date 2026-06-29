from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.helpers.position import Position


class TemplateDetector:
    def __init__(self, template_path: str, threshold: float = 0.85, method: int = cv2.TM_CCOEFF_NORMED) -> None:
        self.template_path = template_path
        self.threshold = threshold
        self.method = method
        self.position = None
        self.seen = None
        self.__post_init__()

    def __post_init__(self) -> None:
        self.template = cv2.imread(
            str(Path(self.template_path)),
            cv2.IMREAD_COLOR,
        )

        if self.template is None:
            raise FileNotFoundError(self.template_path)

        self.template_h, self.template_w = self.template.shape[:2]

    # def score(self, frame: np.ndarray) -> float:
    #     result = cv2.matchTemplate(
    #         frame,
    #         self.template,
    #         self.method,
    #     )
    #
    #
    #     _, score, _, _ = cv2.minMaxLoc(result)
    #
    #     return score

    def detect(self, frame: np.ndarray) -> None:
        result = cv2.matchTemplate(
            frame,
            self.template,
            self.method,
        )

        _, score, _, location = cv2.minMaxLoc(result)

        if score < self.threshold:
            self.position = None
            self.seen = False
            return

        x, y = location

        self.seen = True
        self.position = Position(
            position_x=x + self.template_w // 2,
            position_y=y + self.template_h // 2,
        )

    def distance(self, position: Position) -> float | None:
        return position.distance_to(self.position) if self.seen else None
