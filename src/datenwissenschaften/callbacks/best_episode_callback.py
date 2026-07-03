import os

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.runtime import get_runtime


# noinspection PyMethodMayBeStatic
class BestEpisodeCallback(BaseCallback):
    def __init__(self, total_timesteps: int = 0):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.episodes: list[EpisodeRecord] = []
        self.active_episodes: list[EpisodeRecord] = []
        self.episode_counts: list[int] = []
        self.finished_episode_count = 0
        self.best_time_until_won: int | None = None

    def _on_training_start(self) -> None:
        self._ensure_episode_slots(self.training_env.num_envs)
        runtime = get_runtime()
        self.best_time_until_won = self._previous_time_until_won(runtime.get_state_value("best_time_until_won"))
        logger.info(
            f"Training started across {self.training_env.num_envs} envs. "
            f"Best time until won: {self.best_time_until_won}. Total steps: {self.total_timesteps}"
        )

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

            episode.add_step(infos[env_index])

            if dones[env_index]:
                self._finish_episode(env_index, episode)

        return True

    def _on_rollout_end(self) -> bool:
        if not self.episodes:
            get_runtime().set_state_value("best_episode", "")
            logger.debug("No finished episodes in rollout. won unavailable.")
            return True

        best_episode = self._best_won_episode(self.episodes)
        if best_episode is None:
            get_runtime().set_state_value("best_episode", "")
            logger.debug(f"No won episodes in rollout. Finished episodes: {len(self.episodes)}.")
            self.episodes.clear()
            return True

        runtime = get_runtime()
        previous_best_time_until_won = self._previous_time_until_won(runtime.get_state_value("best_time_until_won"))
        logger.debug(
            f"Rollout best candidate: env={best_episode.env_index}, episode={best_episode.episode_index}, "
            f"won={best_episode.won}, time_until_won={best_episode.time_until_won}, "
            f"previous_best_time_until_won={previous_best_time_until_won}"
        )

        if self._is_new_best(best_episode, previous_best_time_until_won):
            self._save_best_episode(best_episode)
        else:
            runtime.set_state_value("best_episode", "")

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
            f"Env {env_index} finished episode {episode.episode_index}: "
            f"won={episode.won}, time_until_won={episode.time_until_won}"
        )

        self.episode_counts[env_index] += 1
        self.active_episodes[env_index] = EpisodeRecord(env_index, self.episode_counts[env_index])

    def _previous_time_until_won(self, value: str) -> int | None:
        text = str(value).strip()
        if text.isdecimal():
            return int(text)
        return None

    def _is_new_best(self, episode: EpisodeRecord, previous_best_time_until_won: int | None) -> bool:
        if (
            not episode.bk2_path
            or not episode.won
            or episode.time_until_won is None
            or not episode.started_from_initial_savestate
        ):
            return False
        if previous_best_time_until_won is None:
            return True
        return episode.time_until_won < previous_best_time_until_won

    def _save_best_episode(self, episode: EpisodeRecord) -> None:
        self.best_time_until_won = episode.time_until_won
        logger.info(
            f"New best time until won: {episode.time_until_won} " f"({os.path.basename(episode.bk2_path or '')})"
        )

        runtime = get_runtime()
        runtime.set_state_value("best_time_until_won", episode.time_until_won)
        runtime.set_state_value("best_episode", episode.bk2_path)
        runtime.set_state_value(
            "best_episode_started_from_initial_savestate",
            episode.started_from_initial_savestate is True,
        )

    def _bk2_path(self, env_index: int, episode_index: int) -> str:
        runtime = get_runtime()
        filename = f"{runtime.game}-{runtime.savestate}-{episode_index:06d}.bk2"
        return os.path.join(runtime.record_dir, str(env_index), filename)

    def _best_won_episode(self, episodes: list[EpisodeRecord]) -> EpisodeRecord | None:
        won_episodes = [episode for episode in episodes if episode.won and episode.time_until_won is not None]
        if not won_episodes:
            return None
        return min(won_episodes, key=self._time_until_won)

    def _time_until_won(self, episode: EpisodeRecord) -> int:
        if episode.time_until_won is None:
            raise ValueError("Won episode is missing time_until_won.")
        return episode.time_until_won
