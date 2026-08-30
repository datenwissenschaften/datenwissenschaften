from dataclasses import dataclass

import cv2
import numpy as np

from datenwissenschaften.states.detection import BoundingBox


@dataclass(slots=True, frozen=True)
class ColorBlobDetectionConfig:
    colors: tuple[tuple[int, int, int], ...]
    minimum_width: int
    maximum_width: int
    minimum_height: int
    maximum_height: int
    minimum_area: int
    fragment_distance: float
    playfield_width_ratio: float
    playfield_height_ratio: float


def detect_color_blobs(
    frame: np.ndarray,
    config: ColorBlobDetectionConfig,
) -> tuple[tuple[tuple[float, float], ...], tuple[BoundingBox, ...]]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Color blob detection requires an RGB frame")
    playfield_margin = round(frame.shape[1] * (1.0 - config.playfield_width_ratio) / 2.0)
    playfield_height = max(1, round(frame.shape[0] * config.playfield_height_ratio))
    playfield = frame[:playfield_height, playfield_margin : frame.shape[1] - playfield_margin]
    candidates: list[tuple[int, float, float, int, int, int, int]] = []
    for color in config.colors:
        mask = np.all(playfield == color, axis=2).astype(np.uint8)
        _, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates.extend(
            (
                int(area),
                float(center[0] + playfield_margin),
                float(center[1]),
                int(x + playfield_margin),
                int(y),
                int(width),
                int(height),
            )
            for (x, y, width, height, area), center in zip(stats[1:], centers[1:], strict=True)
            if config.minimum_width <= width <= config.maximum_width
            and config.minimum_height <= height <= config.maximum_height
            and area >= config.minimum_area
        )
    candidates.sort(reverse=True)
    selected: list[tuple[float, float]] = []
    boxes: list[BoundingBox] = []
    for _, x, y, left, top, width, height in candidates:
        if all(
            np.hypot(x - selected_x, y - selected_y) >= config.fragment_distance for selected_x, selected_y in selected
        ):
            selected.append((x, y))
            boxes.append((left, top, width, height))
    return tuple(selected), tuple(boxes)
