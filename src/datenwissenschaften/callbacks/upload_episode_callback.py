import json
import os
import platform
import shutil
import subprocess

import requests
from itsdangerous import Signer
from stable_baselines3.common.callbacks import BaseCallback

from loguru import logger
from datenwissenschaften.runtime import get_runtime


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
    def __init__(self):
        super().__init__()
        self.upload_url = os.environ.get("RETRO_SPEEDLAB_UPLOAD_URL") or "https://speedlab.datenwissenschaften.com/api"

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        runtime = get_runtime()
        best_episode_path = runtime.get_state_value("best_episode")

        if not best_episode_path or not os.path.exists(best_episode_path):
            return True

        metadata = runtime.get_model_metadata(self.model)
        metadata["num_timesteps"] = self.num_timesteps
        metadata["game"] = runtime.game
        metadata["savestate"] = runtime.savestate
        metadata["system"] = get_system_metadata()

        secret_key = os.environ.get("RETRO_SPEEDLAB_API_KEY")
        if not secret_key:
            logger.info("RETRO_SPEEDLAB_API_KEY not set. Skipping episode upload.")
            return True

        signing_key = requests.get(
            f"{self.upload_url}/runs/signing-key",
            headers={"X-API-Key": secret_key},
        ).json()["signing_key"]

        signer = Signer(signing_key)
        metadata_json = json.dumps(metadata, indent=4, sort_keys=True)
        signed_metadata = signer.sign(metadata_json.encode("utf-8"))

        try:
            with open(best_episode_path, "rb") as f:
                files = {
                    "bk2_file": (os.path.basename(best_episode_path), f, "application/octet-stream"),
                    "metadata_file": ("metadata.json.signed", signed_metadata, "application/octet-stream"),
                }
                data = {"game": runtime.game, "category": runtime.savestate}
                headers = {
                    "X-API-Key": secret_key,
                }
                logger.info(f"Uploading episode to {self.upload_url}...")
                response = requests.post(f"{self.upload_url}/runs", files=files, data=data, headers=headers, timeout=30)
                response.raise_for_status()
                logger.info("Episode uploaded successfully.")
        except Exception as e:
            logger.error(f"Failed to upload episode: {e}")

        return True
