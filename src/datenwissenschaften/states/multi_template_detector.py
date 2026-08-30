from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.states.detection import BoundingBox


class MultiTemplateDetector:
    def __init__(
        self,
        template_paths: tuple[str | Path, ...],
        threshold: float,
        minimum_distance: float,
    ) -> None:
        if not template_paths:
            raise ValueError("At least one template is required")
        self.templates = tuple(self._load_template(path) for path in template_paths)
        self.threshold = threshold
        self.minimum_distance = minimum_distance
        self.positions: tuple[tuple[float, float], ...] = ()
        self.boxes: tuple[BoundingBox, ...] = ()
        self.seen = False

    @staticmethod
    def _load_template(template_path: str | Path) -> np.ndarray:
        path = Path(template_path).expanduser()
        path = path if path.is_absolute() else Path("assets") / path
        template = cv2.imread(str(path.resolve()), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(path)
        return template

    def detect(self, frame: np.ndarray) -> None:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        candidates: list[tuple[float, int, int, int, int]] = []
        for template in self.templates:
            if any(template > source for template, source in zip(template.shape, image.shape, strict=True)):
                raise ValueError("Template must fit inside the frame")
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.argwhere(result >= self.threshold)
            height, width = template.shape
            candidates.extend((float(result[y, x]), int(x), int(y), width, height) for y, x in locations)
        candidates.sort(reverse=True)
        selected: list[tuple[float, float]] = []
        boxes: list[BoundingBox] = []
        for _, x, y, width, height in candidates:
            center_x = x + width / 2
            center_y = y + height / 2
            if all(
                np.hypot(center_x - selected_x, center_y - selected_y) >= self.minimum_distance
                for selected_x, selected_y in selected
            ):
                selected.append((center_x, center_y))
                boxes.append((x, y, width, height))
        self.positions = tuple(selected)
        self.boxes = tuple(boxes)
        self.seen = bool(self.positions)
