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
    """Learns isolated enemy sprites from RAM-supervised hit events.

    A hit is only supervision for *when* to look.  It is not sufficient evidence
    that an arbitrary rectangle around the actor is an enemy: those rectangles
    mostly contain the scrolling level or HUD.  Candidates therefore have to be
    compact motion components near the actor, and are persisted with a motion
    mask as their alpha channel.
    """

    match_threshold = 0.82
    max_templates = 64

    def __init__(self, state_name: str) -> None:
        self.state_name = state_name
        self.previous_gray: np.ndarray | None = None
        self.previous_hit = False
        self.templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.pending_candidates: dict[str, tuple[np.ndarray, int]] = {}
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
        pending_before_hit = tuple(self.pending_candidates.items())
        for crop in crops:
            descriptor = self._descriptor(crop)
            matching_id = next(
                (
                    candidate_id
                    for candidate_id, (candidate, _count) in pending_before_hit
                    if self._descriptor_similarity(descriptor, candidate) >= 0.88
                ),
                None,
            )
            if matching_id is None:
                self.pending_candidates[self._crop_id(crop)] = (descriptor, 1)
                continue
            candidate, count = self.pending_candidates[matching_id]
            self.pending_candidates[matching_id] = (candidate, count + 1)
            if count < 1:
                continue
            enemy_id = matching_id
            if enemy_id in self.templates:
                continue
            root = self._root()
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{enemy_id}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA)):
                continue
            self.templates[enemy_id] = (self._gray(crop[..., :3]), crop[..., 3])
            learned.append(enemy_id)
            logger.info(f"Learned enemy visual {enemy_id} for Explorer state {self.state_name}")
        if learned:
            self._publish()
        return learned

    def _candidate_crops(self, frame: np.ndarray, gray: np.ndarray, actor: Position) -> list[np.ndarray]:
        height, width = gray.shape
        actor_x = int(np.clip(actor.position_x / max(1, actor.screen_size) * width, 0, width - 1))
        actor_y = int(np.clip(actor.position_y / max(1, actor.screen_size) * height, 0, height - 1))
        if self.previous_gray is None or self.previous_gray.shape != gray.shape:
            return []

        difference = cv2.absdiff(gray, self.previous_gray)
        _, motion_mask = cv2.threshold(difference, 32, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        # A large fraction of the screen changing at once is a camera scroll,
        # transition, or title card—not evidence for an individual sprite.
        if cv2.countNonZero(motion_mask) > width * height * 0.20:
            return []
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            moving_pixels = cv2.countNonZero(motion_mask[y : y + h, x : x + w])
            distance = np.hypot(x + w / 2 - actor_x, y + h / 2 - actor_y)
            compactness = moving_pixels / max(1, box_area)
            contains_actor = x - 2 <= actor_x <= x + w + 2 and y - 2 <= actor_y <= y + h + 2
            if (
                16 <= box_area <= width * height * 0.04
                and w >= 3
                and h >= 3
                and compactness >= 0.18
                and not contains_actor
                and distance <= max(width, height) * 0.25
                and x > 0
                and y > 0
                and x + w < width
                and y + h < height
            ):
                boxes.append((distance, x, y, w, h))

        crops = []
        for _, x, y, w, h in sorted(boxes)[:2]:
            padding = max(2, min(w, h) // 8)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
            rgb = np.asarray(frame[y1:y2, x1:x2, :3], dtype=np.uint8)
            alpha = motion_mask[y1:y2, x1:x2]
            if rgb.size and cv2.countNonZero(alpha) >= 16:
                crops.append(np.dstack((rgb, alpha)))
        return crops

    def _detect(self, gray: np.ndarray) -> list[EnemyDetection]:
        detections = []
        for enemy_id, (template, mask) in tuple(self.templates.items())[: self.max_templates]:
            height, width = template.shape
            if height > gray.shape[0] or width > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED, mask=mask)
            _, score, _, location = cv2.minMaxLoc(result)
            if not np.isfinite(score) or score < self.match_threshold:
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
        self.pending_candidates = {}
        self.previous_gray = None
        self.previous_hit = False
        for path in sorted(root.glob("*.png"))[: self.max_templates]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            # Old candidates were opaque rectangular screenshots.  Never load
            # those as enemies; only the new foreground-masked format is valid.
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                path.unlink(missing_ok=True)
                continue
            alpha = image[..., 3]
            if cv2.countNonZero(alpha) < 16 or cv2.countNonZero(alpha) == alpha.size:
                path.unlink(missing_ok=True)
                continue
            self.templates[path.stem] = (
                cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY),
                alpha,
            )
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

    @staticmethod
    def _descriptor(crop: np.ndarray) -> np.ndarray:
        return cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _descriptor_similarity(first: np.ndarray, second: np.ndarray) -> float:
        first_mask = first[..., 3] >= 64
        second_mask = second[..., 3] >= 64
        union = np.count_nonzero(first_mask | second_mask)
        intersection_mask = first_mask & second_mask
        intersection = np.count_nonzero(intersection_mask)
        if not union or not intersection:
            return 0.0
        shape_similarity = intersection / union
        colors_a = first[..., :3][intersection_mask].astype(np.int16)
        colors_b = second[..., :3][intersection_mask].astype(np.int16)
        color_similarity = 1.0 - float(np.mean(np.abs(colors_a - colors_b)) / 255.0)
        return 0.5 * shape_similarity + 0.5 * color_similarity
