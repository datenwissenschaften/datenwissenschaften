from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from loguru import logger

_MIB = 1024**2
_WORKER_MEMORY_BUDGET = 256 * _MIB
_MIN_MEMORY_RESERVE = 1024 * _MIB


@lru_cache(maxsize=32)
def optimal_env_count(
    worker_limit: int | None = None,
    *,
    cpu_count: int | None = None,
    memory_limit: int | None = None,
) -> int:
    if worker_limit is not None and worker_limit < 1:
        raise ValueError("worker_limit must be positive.")

    detected_cpus = cpu_count if cpu_count is not None else _available_cpu_count()
    detected_memory = memory_limit if memory_limit is not None else _memory_limit()

    cpu_workers = max(1, detected_cpus - 1)
    memory_reserve = max(_MIN_MEMORY_RESERVE, detected_memory // 5)
    memory_workers = max(1, (detected_memory - memory_reserve) // _WORKER_MEMORY_BUDGET)
    candidates = [cpu_workers, memory_workers]
    if worker_limit is not None:
        candidates.append(worker_limit)
    selected = max(1, min(candidates))

    logger.info(
        "Auto-selected environment parallelism: "
        f"num_envs={selected}, cpus={detected_cpus}, "
        f"memory_limit={detected_memory / 1024**3:.1f} GiB"
    )
    return selected


def _available_cpu_count() -> int:
    try:
        available = max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        available = max(1, os.cpu_count() or 1)

    quota = _cpu_quota()
    return min(available, quota) if quota is not None else available


def _cpu_quota() -> int | None:
    try:
        quota_value, period_value = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if quota_value != "max":
            return max(1, int(quota_value) // int(period_value))
    except (OSError, ValueError):
        pass

    quota = _read_integer(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _read_integer(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota is not None and period is not None and quota > 0:
        return max(1, quota // period)
    return None


def _memory_limit() -> int:
    limits = [_host_memory()]

    # cgroup v2 (including systemd MemoryMax).
    cgroup_v2_limit = _read_integer(Path("/sys/fs/cgroup/memory.max"))
    if cgroup_v2_limit is not None:
        limits.append(cgroup_v2_limit)

    # cgroup v1 fallback.
    cgroup_v1_limit = _read_integer(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if cgroup_v1_limit is not None:
        limits.append(cgroup_v1_limit)

    return min(limits)


def _host_memory() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 4 * 1024**3


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value == "max":
            return None
        parsed = int(value)
        # Some cgroup v1 hosts expose an effectively unlimited sentinel.
        return parsed if parsed < 1 << 60 else None
    except (OSError, ValueError):
        return None
