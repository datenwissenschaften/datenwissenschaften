from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

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


@dataclass(frozen=True)
class _VisualComponent:
    bounds: tuple[int, int, int, int]
    appearance: np.ndarray


@dataclass(frozen=True)
class _EnemyTemplate:
    rgb: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    color_histogram: np.ndarray


@dataclass
class _VisualTrack:
    track_id: int
    bounds: tuple[int, int, int, int]
    appearance: np.ndarray
    previous_bounds: tuple[int, int, int, int] | None = None
    hits: int = 1
    age: int = 1
    misses: int = 0
    travel: float = 0.0


class EnemyLearner:
    """Learns actors first, then isolated enemies from supervised hit events.

    Motion components are tracked over time without RAM coordinates.  The actor
    is the uniquely persistent, translating visual track; stationary animated
    scenery cannot qualify.  A hit can only teach an enemy after that actor track
    is confident, and only from the current motion component touching it. A
    candidate becomes detectable and reward-bearing only after a second,
    independent hit produces a matching contact crop.
    """

    match_threshold = 0.88
    color_match_threshold = 0.80
    duplicate_shape_threshold = 0.62
    duplicate_color_threshold = 0.88
    duplicate_appearance_threshold = 0.84
    detection_confirmation_frames = 2
    maximum_detections_per_template = 8
    max_templates = 64
    history_frames = 6
    actor_confirmation_frames = 6
    actor_minimum_age = 8
    track_max_misses = 8
    learner_version = "3"

    def __init__(self, state_name: str) -> None:
        self.state_name = state_name
        self.previous_hit = False
        self.frame_history: deque[np.ndarray] = deque(maxlen=self.history_frames)
        self.visual_tracks: dict[int, _VisualTrack] = {}
        self.actor_track_id: int | None = None
        self._next_track_id = 1
        self.templates: dict[str, _EnemyTemplate] = {}
        self.candidate_templates: dict[str, _EnemyTemplate] = {}
        self._detection_streaks: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}
        self._loaded_root: Path | None = None

    def observe(self, frame: np.ndarray, hit: bool) -> EnemyObservation:
        gray = self._gray(frame)
        self._load_templates()
        self._update_visual_tracks(frame, gray)
        learned = ()
        if hit and not self.previous_hit and self.actor_confident:
            learned = tuple(self._learn_hit_regions(frame, gray))
        detections = tuple(self._detect(frame)) if self.actor_confident else ()
        self.frame_history.append(gray.copy())
        self.previous_hit = hit
        return EnemyObservation(detections=detections, learned_enemy_ids=learned)

    def reset(self) -> None:
        self.frame_history.clear()
        self.previous_hit = False
        self.visual_tracks.clear()
        self.actor_track_id = None
        self._next_track_id = 1
        self._detection_streaks.clear()

    @property
    def actor_confident(self) -> bool:
        return self.actor_bounds is not None

    @property
    def actor_bounds(self) -> tuple[int, int, int, int] | None:
        track = self.visual_tracks.get(self.actor_track_id) if self.actor_track_id is not None else None
        return track.bounds if track is not None and track.misses <= self.track_max_misses else None

    @property
    def actor_center(self) -> tuple[float, float] | None:
        if self.actor_bounds is None:
            return None
        x, y, width, height = self.actor_bounds
        return x + width / 2, y + height / 2

    def _learn_hit_regions(self, frame: np.ndarray, gray: np.ndarray) -> list[str]:
        """Stage first-hit crops and promote them only after an independent matching hit."""
        crops = self._candidate_crops(frame, gray)
        learned = []
        for crop in crops:
            duplicate_id = self._find_duplicate_template(crop)
            if duplicate_id is not None:
                logger.debug(f"Hit revalidated learned enemy visual {duplicate_id}")
                continue
            candidate_id = self._find_duplicate_template(crop, self.candidate_templates)
            if candidate_id is not None:
                if self._promote_candidate(candidate_id):
                    learned.append(candidate_id)
                    logger.info(f"Revalidated game-wide enemy visual {candidate_id}")
                continue
            enemy_id = self._crop_id(crop)
            if enemy_id in self.candidate_templates:
                continue
            root = self._candidate_root()
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{enemy_id}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA)):
                continue
            self.candidate_templates[enemy_id] = self._make_template(crop[..., :3], crop[..., 3])
            logger.info(f"Staged enemy visual {enemy_id}; another matching hit is required")
        if learned:
            self._publish()
        return learned

    def _promote_candidate(self, enemy_id: str) -> bool:
        template = self.candidate_templates.get(enemy_id)
        source = self._candidate_root() / f"{enemy_id}.png"
        if template is None or not source.is_file():
            return False
        root = self._root()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"{enemy_id}.png"
        try:
            source.replace(destination)
        except OSError:
            return False
        self.candidate_templates.pop(enemy_id, None)
        self.templates[enemy_id] = template
        return True

    def _update_visual_tracks(self, frame: np.ndarray, gray: np.ndarray) -> None:
        """Track independently moving sprites and select a confident actor."""
        for track in self.visual_tracks.values():
            track.age += 1
            track.misses += 1

        if not self.frame_history or self.frame_history[-1].shape != gray.shape:
            return
        motion_mask = self._motion_mask(gray, self.frame_history[-1])
        if cv2.countNonZero(motion_mask) > gray.size * 0.20:
            self._expire_tracks()
            return

        components = self._visual_components(frame, motion_mask)
        unmatched_tracks = set(self.visual_tracks)
        unmatched_components = set(range(len(components)))
        matches = []
        for track_id, track in self.visual_tracks.items():
            for component_index, component in enumerate(components):
                distance = self._center_distance(track.bounds, component.bounds)
                old_area = max(1, track.bounds[2] * track.bounds[3])
                new_area = max(1, component.bounds[2] * component.bounds[3])
                size_ratio = max(old_area, new_area) / min(old_area, new_area)
                appearance_distance = 1.0 - float(np.minimum(track.appearance, component.appearance).sum())
                maximum_distance = max(6.0, np.hypot(track.bounds[2], track.bounds[3]) * 1.5)
                if distance <= maximum_distance and size_ratio <= 2.25 and appearance_distance <= 0.65:
                    score = distance + (size_ratio - 1.0) * 4.0 + appearance_distance * 12.0
                    matches.append((score, track_id, component_index))

        for _, track_id, component_index in sorted(matches):
            if track_id not in unmatched_tracks or component_index not in unmatched_components:
                continue
            track = self.visual_tracks[track_id]
            previous_center = self._box_center(track.bounds)
            component = components[component_index]
            track.previous_bounds = track.bounds
            track.bounds = component.bounds
            track.appearance = track.appearance * 0.75 + component.appearance * 0.25
            track.appearance /= max(float(track.appearance.sum()), 1e-6)
            track.travel += float(np.hypot(*(np.subtract(self._box_center(track.bounds), previous_center))))
            track.hits += 1
            track.misses = 0
            unmatched_tracks.remove(track_id)
            unmatched_components.remove(component_index)

        for component_index in unmatched_components:
            track_id = self._next_track_id
            self._next_track_id += 1
            component = components[component_index]
            self.visual_tracks[track_id] = _VisualTrack(track_id, component.bounds, component.appearance)

        self._expire_tracks()
        self._select_actor(gray.shape)

    def _visual_components(self, frame: np.ndarray, motion_mask: np.ndarray) -> list[_VisualComponent]:
        height, width = motion_mask.shape
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        components = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (2 <= w * h <= width * height * 0.04 and x > 0 and y > height * 0.08):
                continue
            padding = max(8, w, h)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
            rgb = np.asarray(frame[y1:y2, x1:x2, :3], dtype=np.uint8)
            alpha = self._isolate_foreground(rgb, motion_mask[y1:y2, x1:x2])
            if cv2.countNonZero(alpha) < 4:
                continue
            ys, xs = np.nonzero(alpha)
            candidate = (
                x1 + int(xs.min()),
                y1 + int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
            if candidate[0] + candidate[2] < width and candidate[1] + candidate[3] < height:
                histogram = cv2.calcHist([rgb], [0, 1, 2], alpha, [4, 4, 4], [0, 256, 0, 256, 0, 256]).ravel()
                histogram /= max(float(histogram.sum()), 1.0)
                components.append(_VisualComponent(candidate, histogram))

        # Different changed edges of one sprite often expand to the same visual
        # object.  Keep one component so it cannot create competing tracks.
        deduplicated = []
        for component in sorted(components, key=lambda item: item.bounds[2] * item.bounds[3], reverse=True):
            if all(self._intersection_over_union(component.bounds, other.bounds) < 0.65 for other in deduplicated):
                deduplicated.append(component)
        return deduplicated

    def _select_actor(self, frame_shape: tuple[int, int]) -> None:
        minimum_travel = min(frame_shape) * 0.04
        eligible = [
            track
            for track in self.visual_tracks.values()
            if track.hits >= self.actor_confirmation_frames
            and track.age >= self.actor_minimum_age
            and track.travel >= minimum_travel
            and track.misses <= 2
        ]
        if not eligible:
            return
        ranked = sorted(eligible, key=self._actor_track_score, reverse=True)
        current = self.visual_tracks.get(self.actor_track_id) if self.actor_track_id is not None else None
        if current is not None and any(track.track_id == current.track_id for track in eligible):
            return
        runner_up_score = self._actor_track_score(ranked[1]) if len(ranked) > 1 else float("-inf")
        if self._actor_track_score(ranked[0]) >= runner_up_score + 3.0:
            self.actor_track_id = ranked[0].track_id

    @staticmethod
    def _actor_track_score(track: _VisualTrack) -> float:
        continuity = track.hits / max(1, track.age)
        return track.hits + track.age * 0.25 + continuity * 4.0 + min(track.travel, 64.0) * 0.05 - track.misses * 2.0

    def _expire_tracks(self) -> None:
        expired = [track_id for track_id, track in self.visual_tracks.items() if track.misses > self.track_max_misses]
        for track_id in expired:
            self.visual_tracks.pop(track_id, None)
            if track_id == self.actor_track_id:
                self.actor_track_id = None

    def _candidate_crops(self, frame: np.ndarray, gray: np.ndarray) -> list[np.ndarray]:
        height, width = gray.shape
        if self.actor_bounds is None or not self.frame_history or self.frame_history[-1].shape != gray.shape:
            return []

        motion_mask = self._motion_mask(gray, self.frame_history[-1])
        # A large fraction of the screen changing at once is a camera scroll,
        # transition, or title card—not evidence for an individual sprite.
        if cv2.countNonZero(motion_mask) > width * height * 0.20:
            return []

        actor_bounds = self.actor_bounds
        if actor_bounds is None:
            return []
        actor_x, actor_y, actor_w, actor_h = actor_bounds
        # The actor is also moving at a collision.  Remove its learned visual
        # envelope before finding the other object involved in that collision.
        motion_mask[actor_y : actor_y + actor_h, actor_x : actor_x + actor_w] = 0
        actor_track = self.visual_tracks.get(self.actor_track_id) if self.actor_track_id is not None else None
        if actor_track is not None and actor_track.previous_bounds is not None:
            previous_x, previous_y, previous_w, previous_h = actor_track.previous_bounds
            motion_mask[previous_y : previous_y + previous_h, previous_x : previous_x + previous_w] = 0
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        contact_tolerance = max(2.0, min(actor_w, actor_h) * 0.25)
        actor_area = max(1, actor_w * actor_h)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # An animation can change only an interior pixel.  Expand the seed
            # into its complete object before deciding whether it made contact.
            if self._box_distance(actor_bounds, (x, y, w, h)) > max(actor_w, actor_h) * 1.5:
                continue
            padding = max(8, w, h)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
            rgb = np.asarray(frame[y1:y2, x1:x2, :3], dtype=np.uint8)
            alpha = self._isolate_foreground(rgb, motion_mask[y1:y2, x1:x2])
            ax1, ay1 = max(0, actor_x - x1), max(0, actor_y - y1)
            ax2 = min(alpha.shape[1], actor_x + actor_w - x1)
            ay2 = min(alpha.shape[0], actor_y + actor_h - y1)
            alpha[ay1:ay2, ax1:ax2] = 0
            if cv2.countNonZero(alpha) < 16:
                continue
            ys, xs = np.nonzero(alpha)
            trim_x1, trim_x2 = max(0, int(xs.min()) - 1), min(alpha.shape[1], int(xs.max()) + 2)
            trim_y1, trim_y2 = max(0, int(ys.min()) - 1), min(alpha.shape[0], int(ys.max()) + 2)
            object_bounds = (
                x1 + int(xs.min()),
                y1 + int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
            box_area = object_bounds[2] * object_bounds[3]
            contact_gap = self._box_distance(actor_bounds, object_bounds)
            size_difference = abs(np.log(max(1, box_area) / actor_area))
            if (
                1 <= box_area <= actor_area * 3.0
                and contact_gap <= contact_tolerance
                and object_bounds[0] > 0
                and object_bounds[1] > 0
                and object_bounds[0] + object_bounds[2] < width
                and object_bounds[1] + object_bounds[3] < height
            ):
                crop = np.dstack((rgb, alpha))[trim_y1:trim_y2, trim_x1:trim_x2]
                candidates.append((contact_gap, size_difference, crop))
        return [min(candidates, key=lambda candidate: candidate[:2])[2]] if candidates else []

    @staticmethod
    def _motion_mask(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(cv2.absdiff(current, previous), 12, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    @staticmethod
    def _center_distance(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        return float(np.hypot(*(np.subtract(EnemyLearner._box_center(first), EnemyLearner._box_center(second)))))

    @staticmethod
    def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, width, height = box
        return x + width / 2, y + height / 2

    @staticmethod
    def _intersection_over_union(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        intersection_width = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        intersection_height = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = intersection_width * intersection_height
        return intersection / max(1, w1 * h1 + w2 * h2 - intersection)

    @staticmethod
    def _box_distance(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        return float(np.hypot(max(x1 - (x2 + w2), x2 - (x1 + w1), 0), max(y1 - (y2 + h2), y2 - (y1 + h1), 0)))

    @staticmethod
    def _isolate_foreground(rgb: np.ndarray, motion_seed: np.ndarray) -> np.ndarray:
        """Expand changed pixels into the complete current-frame sprite."""
        if not rgb.size or min(rgb.shape[:2]) < 3:
            return motion_seed

        border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
        background_color = np.median(border.astype(np.float32), axis=0)
        color_distance = np.linalg.norm(rgb.astype(np.float32) - background_color, axis=2)
        foreground_seed = (motion_seed > 0) & (color_distance >= 24.0)
        if np.count_nonzero(foreground_seed) < 2:
            foreground_seed = motion_seed > 0

        grabcut_mask = np.full(rgb.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        grabcut_mask[color_distance >= 24.0] = cv2.GC_PR_FGD
        grabcut_mask[[0, -1], :] = cv2.GC_BGD
        grabcut_mask[:, [0, -1]] = cv2.GC_BGD
        grabcut_mask[foreground_seed] = cv2.GC_FGD
        try:
            cv2.grabCut(
                rgb,
                grabcut_mask,
                None,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                3,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return cv2.dilate(motion_seed, np.ones((3, 3), np.uint8))

        foreground = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        component_count, labels = cv2.connectedComponents(foreground)
        touching_labels = set(int(label) for label in labels[foreground_seed] if label)
        if component_count <= 1 or not touching_labels:
            return cv2.dilate(motion_seed, np.ones((3, 3), np.uint8))
        isolated = np.isin(labels, tuple(touching_labels)).astype(np.uint8) * 255
        return cv2.morphologyEx(isolated, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def _detect(self, frame: np.ndarray) -> list[EnemyDetection]:
        """Return only spatially and temporally consistent color-template matches."""
        rgb = np.asarray(frame[..., :3], dtype=np.uint8)
        gray = self._gray(rgb)
        raw_detections = []
        actor_bounds = self.actor_bounds
        for enemy_id, template in tuple(self.templates.items())[: self.max_templates]:
            height, width = template.gray.shape
            if height > gray.shape[0] or width > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template.gray, cv2.TM_CCORR_NORMED, mask=template.mask)
            finite_result = np.where(np.isfinite(result), result, -1.0)
            local_maxima = finite_result == cv2.dilate(finite_result, np.ones((3, 3), np.float32))
            locations = np.argwhere(local_maxima & (finite_result >= self.match_threshold))
            ranked_locations = sorted(locations, key=lambda point: finite_result[tuple(point)], reverse=True)
            for y, x in ranked_locations[: self.maximum_detections_per_template]:
                bounds = (int(x), int(y), width, height)
                if actor_bounds is not None and self._intersection_over_union(bounds, actor_bounds) > 0.10:
                    continue
                patch = rgb[y : y + height, x : x + width]
                color_score = cv2.compareHist(
                    template.color_histogram,
                    self._color_histogram(patch, template.mask),
                    cv2.HISTCMP_CORREL,
                )
                if not np.isfinite(color_score) or color_score < self.color_match_threshold:
                    continue
                gray_score = float(finite_result[y, x])
                raw_detections.append(
                    EnemyDetection(
                        enemy_id=enemy_id,
                        score=float(np.clip(gray_score * 0.65 + color_score * 0.35, 0.0, 1.0)),
                        center=(int(x) + width // 2, int(y) + height // 2),
                        bounds=bounds,
                    )
                )

        # Multiple learned animation frames can still match the same object.
        # Greedy NMS exposes one detection, independent of how many templates exist.
        suppressed = []
        for detection in sorted(raw_detections, key=lambda item: item.score, reverse=True):
            if all(self._intersection_over_union(detection.bounds, kept.bounds) < 0.35 for kept in suppressed):
                suppressed.append(detection)

        confirmed = []
        next_streaks: dict[str, list[tuple[tuple[int, int, int, int], int]]] = {}
        for detection in suppressed:
            previous = self._detection_streaks.get(detection.enemy_id, [])
            matching_streaks = [
                streak
                for bounds, streak in previous
                if self._center_distance(bounds, detection.bounds)
                <= max(4.0, np.hypot(detection.bounds[2], detection.bounds[3]))
            ]
            streak = max(matching_streaks, default=0) + 1
            next_streaks.setdefault(detection.enemy_id, []).append((detection.bounds, streak))
            if streak >= self.detection_confirmation_frames:
                confirmed.append(detection)
        self._detection_streaks = next_streaks
        return confirmed

    def _load_templates(self) -> None:
        root = self._root()
        if root == self._loaded_root:
            return
        root.mkdir(parents=True, exist_ok=True)
        version_file = root / ".learner-version"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except OSError:
            version = ""
        if version == "2":
            # Version 2 templates had one supervised hit. Keep them as
            # provisional evidence, but require the new independent contact hit
            # before they can drive observations or rewards again.
            candidate_root = self._candidate_root()
            candidate_root.mkdir(parents=True, exist_ok=True)
            for path in root.glob("*.png"):
                destination = candidate_root / path.name
                suffix = 1
                while destination.exists():
                    destination = candidate_root / f"{path.stem}-{suffix}{path.suffix}"
                    suffix += 1
                path.replace(destination)
            version_file.write_text(self.learner_version, encoding="utf-8")
        elif version != self.learner_version:
            # Earlier files were learned without a confirmed actor/contact and
            # cannot be distinguished reliably from animated scenery.
            for path in root.glob("*.png"):
                quarantine = path.with_suffix(f"{path.suffix}.unverified")
                suffix = 1
                while quarantine.exists():
                    quarantine = path.with_suffix(f"{path.suffix}.unverified-{suffix}")
                    suffix += 1
                path.replace(quarantine)
            version_file.write_text(self.learner_version, encoding="utf-8")
        self.templates = {}
        self.candidate_templates = {}
        self.frame_history.clear()
        self.previous_hit = False
        self.visual_tracks.clear()
        self.actor_track_id = None
        self._next_track_id = 1
        self._detection_streaks.clear()
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
            rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
            candidate = np.dstack((rgb, alpha))
            duplicate_id = self._find_duplicate_template(candidate)
            if duplicate_id is not None:
                self._quarantine_duplicate(path, duplicate_id)
                continue
            self.templates[path.stem] = self._make_template(rgb, alpha)
        candidate_root = self._candidate_root()
        candidate_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(candidate_root.glob("*.png"))[: self.max_templates]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                path.unlink(missing_ok=True)
                continue
            alpha = image[..., 3]
            if cv2.countNonZero(alpha) < 16 or cv2.countNonZero(alpha) == alpha.size:
                path.unlink(missing_ok=True)
                continue
            rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
            candidate = np.dstack((rgb, alpha))
            if self._find_duplicate_template(candidate) is not None:
                self._quarantine_duplicate(path, "verified")
                continue
            duplicate_id = self._find_duplicate_template(candidate, self.candidate_templates)
            if duplicate_id is not None:
                self._quarantine_duplicate(path, duplicate_id)
                continue
            self.candidate_templates[path.stem] = self._make_template(rgb, alpha)
        self._loaded_root = root

    def _find_duplicate_template(
        self,
        crop: np.ndarray,
        templates: dict[str, _EnemyTemplate] | None = None,
    ) -> str | None:
        """Find the same sprite despite small crop, palette, or animation differences."""
        candidate_rgb = np.asarray(crop[..., :3], dtype=np.uint8)
        candidate_mask = np.asarray(crop[..., 3], dtype=np.uint8)
        size = (24, 24)
        candidate_gray = cv2.resize(self._gray(candidate_rgb), size, interpolation=cv2.INTER_AREA)
        candidate_alpha = cv2.resize(candidate_mask, size, interpolation=cv2.INTER_NEAREST) > 0
        candidate_histogram = self._color_histogram(candidate_rgb, candidate_mask)
        for enemy_id, template in (self.templates if templates is None else templates).items():
            known_gray = cv2.resize(template.gray, size, interpolation=cv2.INTER_AREA)
            known_alpha = cv2.resize(template.mask, size, interpolation=cv2.INTER_NEAREST) > 0
            intersection = candidate_alpha & known_alpha
            union = candidate_alpha | known_alpha
            shape_score = float(np.count_nonzero(intersection) / max(1, np.count_nonzero(union)))
            if shape_score < self.duplicate_shape_threshold or np.count_nonzero(intersection) < 16:
                continue
            comparison_mask = intersection.astype(np.uint8) * 255
            appearance_score = float(
                cv2.matchTemplate(candidate_gray, known_gray, cv2.TM_CCORR_NORMED, mask=comparison_mask)[0, 0]
            )
            color_score = float(cv2.compareHist(candidate_histogram, template.color_histogram, cv2.HISTCMP_CORREL))
            if (
                np.isfinite(appearance_score)
                and np.isfinite(color_score)
                and appearance_score >= self.duplicate_appearance_threshold
                and color_score >= self.duplicate_color_threshold
            ):
                return enemy_id
        return None

    @classmethod
    def _make_template(cls, rgb: np.ndarray, mask: np.ndarray) -> _EnemyTemplate:
        rgb = np.asarray(rgb, dtype=np.uint8).copy()
        mask = np.asarray(mask, dtype=np.uint8).copy()
        return _EnemyTemplate(rgb, cls._gray(rgb), mask, cls._color_histogram(rgb, mask))

    @staticmethod
    def _color_histogram(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        histogram = cv2.calcHist([rgb], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1).ravel()

    @staticmethod
    def _quarantine_duplicate(path: Path, duplicate_id: str) -> None:
        quarantine = path.with_suffix(f"{path.suffix}.duplicate-of-{duplicate_id}")
        suffix = 1
        while quarantine.exists():
            quarantine = path.with_suffix(f"{path.suffix}.duplicate-of-{duplicate_id}-{suffix}")
            suffix += 1
        path.replace(quarantine)

    def _root(self) -> Path:
        runtime = get_runtime()
        return runtime.cache_dir / "learned_enemies" / runtime.game

    def _candidate_root(self) -> Path:
        return self._root() / ".candidates"

    def _publish(self) -> None:
        publish_metadata(
            "learned_enemies",
            {
                self.state_name: {
                    "count": len(self.templates),
                    "scope": "game",
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
