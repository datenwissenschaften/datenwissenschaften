from pathlib import Path

import cv2
import numpy as np


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
        result = cv2.matchTemplate(image, self.template, cv2.TM_SQDIFF_NORMED)
        score, _, location, _ = cv2.minMaxLoc(result)
        tolerance = np.finfo(result.dtype).eps * self.template.size
        self.seen = score <= tolerance
        if self.seen:
            height, width = self.template.shape
            self.position = location[0] + width / 2, location[1] + height / 2
        else:
            self.position = None
