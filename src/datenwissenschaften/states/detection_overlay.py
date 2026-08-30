from dataclasses import dataclass

import cv2
import numpy as np

from datenwissenschaften.states.detection import BoundingBox

ACTIVE_BOX_COLOR = (255, 255, 0)
INACTIVE_BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
FONT_SCALE = 0.3


@dataclass(slots=True, frozen=True)
class Detection:
    box: BoundingBox
    label: str
    active: bool


def draw_detections(frame: np.ndarray, detections: tuple[Detection, ...]) -> np.ndarray:
    for detection in detections:
        left, top, width, height = detection.box
        color = ACTIVE_BOX_COLOR if detection.active else INACTIVE_BOX_COLOR
        cv2.rectangle(frame, (left, top), (left + width, top + height), color, 1)
        cv2.putText(
            frame,
            detection.label,
            (left, max(top - 3, 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return frame
