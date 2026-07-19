from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from datenwissenschaften.runtime import get_runtime

if TYPE_CHECKING:
    from datenwissenschaften.callbacks.episode_record import EpisodeRecord


def record_rollout_videos(episodes: list[EpisodeRecord], rollout: int) -> list[Path]:
    """Render the highest-scoring completed episode for each curriculum in a rollout."""
    best_by_curriculum: dict[str, EpisodeRecord] = {}
    runtime = get_runtime()
    for episode in episodes:
        curriculum = episode.curriculum_state or runtime.savestate or "default"
        incumbent = best_by_curriculum.get(curriculum)
        candidate_rank = (episode.curriculum_succeeded, episode.score, -episode.step_count)
        incumbent_rank = (
            (incumbent.curriculum_succeeded, incumbent.score, -incumbent.step_count) if incumbent is not None else None
        )
        if incumbent_rank is None or candidate_rank > incumbent_rank:
            best_by_curriculum[curriculum] = episode

    videos = []
    for curriculum, episode in best_by_curriculum.items():
        source = _resolve_recording(episode.bk2_path, runtime.record_dir)
        if source is None:
            logger.warning(f"Cannot render rollout video; episode recording is missing: {episode.bk2_path}")
            continue
        video = source.with_suffix(".mp4")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "datenwissenschaften.rollout_video_playback",
                    "--roms-dir",
                    str(runtime.paths.roms_path),
                    "--no-audio",
                    str(source),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not video.is_file():
                raise FileNotFoundError(f"Movie playback did not create {video}")
            metadata = {
                "game": runtime.game,
                "savestate": runtime.savestate,
                "curriculum": curriculum,
                "rollout": rollout,
                "score": episode.score,
                "steps": episode.step_count,
                "won": episode.won,
                "curriculum_succeeded": episode.curriculum_succeeded,
                "video": video.name,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            video.with_suffix(".rollout.json").write_text(json.dumps(metadata), encoding="utf-8")
            videos.append(video)
            logger.info(
                f"Recorded rollout {rollout} best episode for {curriculum}: "
                f"score={episode.score:g}, video={video.name}"
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            video.unlink(missing_ok=True)
            details = (
                error.stderr.strip()
                if isinstance(error, subprocess.CalledProcessError) and error.stderr
                else str(error)
            )
            logger.warning(f"Could not render rollout video from {source.name}: {details}")
    return videos


def _resolve_recording(requested_path: str, record_root: Path) -> Path | None:
    requested = Path(requested_path)
    if requested.is_file():
        return requested
    # Preserve the worker directory when resolving legacy callback paths.
    # Episode counters are local to each worker, so matching only by filename
    # can silently render another environment's movie with the selected score.
    matches = list(record_root.glob(f"**/{requested.parent.name}/{requested.name}"))
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None
