import os

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.rollout_video import record_rollout_videos
from datenwissenschaften.runtime import get_runtime


# noinspection PyMethodMayBeStatic
class BestEpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes: list[EpisodeRecord] = []
        self.active_episodes: list[EpisodeRecord] = []
        self.episode_counts: list[int] = []
        self.finished_episode_count = 0
        self.rollout_count = 0

    def _on_training_start(self) -> None:
        self._ensure_episode_slots(self.training_env.num_envs)
        logger.info(f"Training started across {self.training_env.num_envs} envs")

    def _on_training_end(self) -> None:
        logger.info("Training stopped")

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if rewards is None or dones is None or infos is None:
            return True

        self._ensure_episode_slots(len(rewards))

        for env_index in range(len(rewards)):
            episode = self.active_episodes[env_index]

            episode.add_step(infos[env_index], rewards[env_index])

            if dones[env_index]:
                self._finish_episode(env_index, episode)

        return True

    def _on_rollout_end(self) -> bool:
        self.rollout_count += 1
        record_rollout_videos(self.episodes, self.rollout_count)
        self.episodes.clear()
        return True

    def _ensure_episode_slots(self, count: int) -> None:
        while len(self.active_episodes) < count:
            env_index = len(self.active_episodes)
            self.active_episodes.append(EpisodeRecord(env_index, 0))
            self.episode_counts.append(0)

    def _finish_episode(self, env_index: int, episode: EpisodeRecord) -> None:
        episode.bk2_path = self._bk2_path(env_index, episode.episode_index)

        self.episodes.append(episode.clone())
        self.finished_episode_count += 1
        logger.debug(
            f"Env {env_index} finished episode {episode.episode_index}: " f"won={episode.won}, score={episode.score:g}"
        )

        self.episode_counts[env_index] += 1
        self.active_episodes[env_index] = EpisodeRecord(env_index, self.episode_counts[env_index])

    def _bk2_path(self, env_index: int, episode_index: int) -> str:
        runtime = get_runtime()
        filename = f"{runtime.game}-{runtime.savestate}-{episode_index:06d}.bk2"
        return os.path.join(runtime.record_dir, runtime.game, runtime.savestate, str(env_index), filename)
