from pathlib import Path

import cv2
import numpy as np

MATCH_THRESHOLD = 0.85
MINIMUM_TARGET_DISTANCE = 4.0


class TemplateDetector:
    def __init__(self, template_path: str | Path) -> None:
        path = Path(template_path).expanduser()
        path = path if path.is_absolute() else Path("assets") / path
        self.template = cv2.imread(str(path.resolve()), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(path)
        self.seen = False
        self.position: tuple[float, float] | None = None

    def detect(self, frame: np.ndarray) -> None:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        if any(template > source for template, source in zip(self.template.shape, image.shape, strict=True)):
            raise ValueError("Template must fit inside the frame")
        result = cv2.matchTemplate(image, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        self.seen = score >= MATCH_THRESHOLD
        if self.seen:
            height, width = self.template.shape
            self.position = location[0] + width / 2, location[1] + height / 2
        else:
            self.position = None


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
        candidates: list[tuple[float, float, float]] = []
        for template in self.templates:
            if any(template > source for template, source in zip(template.shape, image.shape, strict=True)):
                raise ValueError("Template must fit inside the frame")
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.argwhere(result >= self.threshold)
            height, width = template.shape
            candidates.extend((float(result[y, x]), x + width / 2, y + height / 2) for y, x in locations)
        candidates.sort(reverse=True)
        selected: list[tuple[float, float]] = []
        for _, x, y in candidates:
            if all(
                np.hypot(x - selected_x, y - selected_y) >= self.minimum_distance for selected_x, selected_y in selected
            ):
                selected.append((x, y))
        self.positions = tuple(selected)
        self.seen = bool(self.positions)
