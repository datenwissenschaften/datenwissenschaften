import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import requests
from itsdangerous import Signer
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.serialization import to_json_value
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, UploadSettings, load_config


def get_system_metadata() -> dict:
    total_memory_bytes = _get_total_memory_bytes()

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "name": _get_cpu_name(),
            "processor": platform.processor(),
            "logical_cores": os.cpu_count(),
        },
        "memory": {
            "total_bytes": total_memory_bytes,
            "total_gib": round(total_memory_bytes / 1024**3, 2) if total_memory_bytes else None,
        },
        "gpu": _get_gpu_metadata(),
    }


def _get_cpu_name() -> str | None:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    processor = platform.processor()
    return processor or None


def _get_total_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size
        except (OSError, ValueError):
            pass

    return None


def _get_gpu_metadata() -> dict:
    metadata = {
        "cuda_available": False,
        "devices": [],
    }

    try:
        import torch

        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["torch_version"] = torch.__version__

        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            metadata["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "major": properties.major,
                    "minor": properties.minor,
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
    except Exception as error:
        metadata["torch_error"] = str(error)

    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        try:
            result = subprocess.run(
                [
                    nvidia_smi_path,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            metadata["nvidia_smi"] = [
                _parse_nvidia_smi_gpu(line) for line in result.stdout.splitlines() if line.strip()
            ]
        except Exception as error:
            metadata["nvidia_smi_error"] = str(error)

    return metadata


def _parse_nvidia_smi_gpu(line: str) -> dict:
    name, memory_total_mb, driver_version = [part.strip() for part in line.split(",", 2)]
    return {
        "name": name,
        "memory_total_mb": int(memory_total_mb),
        "driver_version": driver_version,
    }


class UploadEpisodeCallback(BaseCallback):
    def __init__(
        self,
        settings: UploadSettings | None = None,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        super().__init__()
        self.settings = load_config(config_path).upload if settings is None else settings
        self.upload_url = self.settings.url
        self.successful_episodes: list[EpisodeRecord] = []
        self.active_episodes: list[EpisodeRecord] = []
        self.episode_counts: list[int] = []

    def _on_step(self):
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

    def _on_rollout_end(self):
        runtime = get_runtime()
        episodes = self.successful_episodes
        self.successful_episodes = []
        if not episodes:
            return True

        metadata = dict(runtime.get_model_metadata(self.model))
        metadata["num_timesteps"] = self.num_timesteps
        metadata["game"] = runtime.game
        metadata["savestate"] = runtime.savestate
        metadata["system"] = get_system_metadata()

        secret_key = self.settings.api_key
        if not secret_key:
            logger.info("Upload API key is not configured. Skipping episode upload.")
            return True

        try:
            signing_key = requests.get(
                f"{self.upload_url}/runs/signing-key",
                headers={"X-API-Key": secret_key},
            ).json()["signing_key"]
        except Exception as e:
            logger.error(f"Failed to get upload signing key: {e}")
            return True

        signer = Signer(signing_key)
        metadata_json = json.dumps(to_json_value(metadata), indent=4, sort_keys=True)
        signed_metadata = signer.sign(metadata_json.encode("utf-8"))

        for episode in episodes:
            self._upload_episode(episode.bk2_path, signed_metadata, secret_key, runtime)

        return True

    def _ensure_episode_slots(self, count: int) -> None:
        while len(self.active_episodes) < count:
            env_index = len(self.active_episodes)
            self.active_episodes.append(EpisodeRecord(env_index, 0))
            self.episode_counts.append(0)

    def _finish_episode(self, env_index: int, episode: EpisodeRecord) -> None:
        runtime = get_runtime()
        filename = f"{runtime.game}-{runtime.savestate}-{episode.episode_index:06d}.bk2"
        episode.bk2_path = os.path.join(runtime.record_dir, str(env_index), filename)

        if episode.won:
            self.successful_episodes.append(episode.clone())

        self.episode_counts[env_index] += 1
        self.active_episodes[env_index] = EpisodeRecord(env_index, self.episode_counts[env_index])

    def _upload_episode(self, episode_path: str, signed_metadata: bytes, secret_key: str, runtime) -> None:
        if not os.path.exists(episode_path):
            logger.error(f"Successful episode recording does not exist: {episode_path}")
            return

        try:
            with open(episode_path, "rb") as episode_file:
                files = {
                    "bk2_file": (os.path.basename(episode_path), episode_file, "application/octet-stream"),
                    "metadata_file": ("metadata.json.signed", signed_metadata, "application/octet-stream"),
                }
                data = {"game": runtime.game, "category": runtime.savestate}
                headers = {"X-API-Key": secret_key}
                logger.info(f"Uploading episode {os.path.basename(episode_path)} to {self.upload_url}...")
                response = requests.post(
                    f"{self.upload_url}/runs",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                logger.info(f"Episode {os.path.basename(episode_path)} uploaded successfully.")
        except Exception as e:
            logger.error(f"Failed to upload episode {os.path.basename(episode_path)}: {e}")
