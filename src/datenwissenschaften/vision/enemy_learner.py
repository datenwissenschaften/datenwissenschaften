from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from datenwissenschaften.helpers.position import Position
from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.ui import publish_metadata


@dataclass(frozen=True)
class EnemyDetection:
    enemy_id: str
    score: float
    center: tuple[int, int]
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class EnemyObservation:
    detections: tuple[EnemyDetection, ...]
    learned_enemy_ids: tuple[str, ...]


class EnemyLearner:
    """Learns reusable enemy crops from RAM-supervised hit events."""

    match_threshold = 0.82
    max_templates = 64

    def __init__(self, state_name: str) -> None:
        self.state_name = state_name
        self.previous_gray: np.ndarray | None = None
        self.previous_hit = False
        self.templates: dict[str, np.ndarray] = {}
        self._loaded_root: Path | None = None

    def observe(self, frame: np.ndarray, actor: Position, hit: bool) -> EnemyObservation:
        gray = self._gray(frame)
        self._load_templates()
        learned = ()
        if hit and not self.previous_hit:
            learned = tuple(self._learn_hit_regions(frame, gray, actor))
        detections = tuple(self._detect(gray))
        self.previous_gray = gray
        self.previous_hit = hit
        return EnemyObservation(detections=detections, learned_enemy_ids=learned)

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_hit = False

    def _learn_hit_regions(self, frame: np.ndarray, gray: np.ndarray, actor: Position) -> list[str]:
        crops = self._candidate_crops(frame, gray, actor)
        learned = []
        for crop in crops:
            enemy_id = self._crop_id(crop)
            if enemy_id in self.templates:
                continue
            root = self._root()
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{enemy_id}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                continue
            self.templates[enemy_id] = self._gray(crop)
            learned.append(enemy_id)
            logger.info(f"Learned enemy visual {enemy_id} for Explorer state {self.state_name}")
        if learned:
            self._publish()
        return learned

    def _candidate_crops(self, frame: np.ndarray, gray: np.ndarray, actor: Position) -> list[np.ndarray]:
        height, width = gray.shape
        actor_x = int(np.clip(actor.position_x / max(1, actor.screen_size) * width, 0, width - 1))
        actor_y = int(np.clip(actor.position_y / max(1, actor.screen_size) * height, 0, height - 1))
        boxes = []
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape:
            difference = cv2.absdiff(gray, self.previous_gray)
            _, mask = cv2.threshold(difference, 24, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                distance = np.hypot(x + w / 2 - actor_x, y + h / 2 - actor_y)
                if 16 <= area <= width * height * 0.12 and distance <= max(width, height) * 0.3:
                    boxes.append((distance, x, y, w, h))

        if not boxes:
            size = max(16, min(width, height) // 5)
            boxes = [(0.0, actor_x - size // 2, actor_y - size // 2, size, size)]

        crops = []
        for _, x, y, w, h in sorted(boxes)[:3]:
            padding = max(2, min(w, h) // 6)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                crops.append(np.asarray(crop, dtype=np.uint8))
        return crops

    def _detect(self, gray: np.ndarray) -> list[EnemyDetection]:
        detections = []
        for enemy_id, template in tuple(self.templates.items())[: self.max_templates]:
            height, width = template.shape
            if height > gray.shape[0] or width > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            if score < self.match_threshold:
                continue
            x, y = location
            detections.append(
                EnemyDetection(
                    enemy_id=enemy_id,
                    score=float(score),
                    center=(x + width // 2, y + height // 2),
                    bounds=(x, y, width, height),
                )
            )
        return detections

    def _load_templates(self) -> None:
        root = self._root()
        if root == self._loaded_root:
            return
        self.templates = {}
        for path in sorted(root.glob("*.png"))[: self.max_templates]:
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.templates[path.stem] = template
        self._loaded_root = root

    def _root(self) -> Path:
        runtime = get_runtime()
        return runtime.cache_dir / "learned_enemies" / runtime.game / runtime.savestate / self.state_name

    def _publish(self) -> None:
        publish_metadata(
            "learned_enemies",
            {
                self.state_name: {
                    "count": len(self.templates),
                    "savestate": get_runtime().savestate,
                }
            },
        )

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else np.asarray(frame, dtype=np.uint8)

    @staticmethod
    def _crop_id(crop: np.ndarray) -> str:
        normalized = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
        return hashlib.sha256(normalized.tobytes()).hexdigest()[:16]
