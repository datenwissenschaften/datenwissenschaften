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
        source = _resolve_recording(
            episode.bk2_path,
            runtime.record_dir,
            game=runtime.game,
            savestate=runtime.savestate,
            env_index=episode.env_index,
        )
        if source is None:
            logger.warning(
                f"Cannot render rollout video; exact episode recording is missing or belongs to another worker: "
                f"{episode.bk2_path or '<not reported>'}"
            )
            continue
        video = source.with_suffix(".mp4")
        metadata_path = video.with_suffix(".rollout.json")
        metadata_temp = metadata_path.with_name(f"{metadata_path.name}.tmp")
        try:
            # Never accept output left behind by an earlier process or a failed
            # render of a BK2 with the same Stable Retro movie counter.
            video.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
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
                "episode_start_state": episode.episode_start_state,
                "environment": episode.env_index,
                "episode": episode.episode_index,
                "recording": source.name,
                "rollout": rollout,
                "score": episode.score,
                "steps": episode.step_count,
                "won": episode.won,
                "curriculum_succeeded": episode.curriculum_succeeded,
                "curriculum_mastered": episode.curriculum_mastered,
                "video": video.name,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            metadata_temp.write_text(json.dumps(metadata), encoding="utf-8")
            metadata_temp.replace(metadata_path)
            videos.append(video)
            logger.info(
                f"Recorded rollout {rollout} best episode for {curriculum}: score={episode.score:g}, video={video.name}"
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            video.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
            details = (
                error.stderr.strip()
                if isinstance(error, subprocess.CalledProcessError) and error.stderr
                else str(error)
            )
            logger.warning(f"Could not render rollout video from {source.name}: {details}")
    return videos


def _resolve_recording(
    requested_path: str,
    record_root: Path,
    *,
    game: str,
    savestate: str,
    env_index: int,
) -> Path | None:
    """Resolve only the BK2 owned by the episode's exact vector worker."""
    if not requested_path:
        return None

    requested = Path(requested_path)
    worker_dir = (Path(record_root) / game / savestate / str(env_index)).resolve()
    candidates = (requested, worker_dir / requested.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.parent == worker_dir and resolved.is_file():
            return resolved
    return None
