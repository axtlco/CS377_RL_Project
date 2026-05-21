from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from .seeding import SeedStream


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_run_metadata(cfg: Any | None = None, device: torch.device | str | None = None) -> dict[str, Any]:
    seed = int(getattr(cfg, "seed", 0)) if cfg is not None and hasattr(cfg, "seed") else 0
    stream = SeedStream(seed)
    git_status = _git_output(["status", "--short"])
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": str(device) if device is not None else str(getattr(cfg, "device", "unknown")),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status_short": git_status or "",
        "seed": seed,
        "seed_protocol": {
            "stride": stream.stride,
            "training_episode_seed": "seed * stride + episode_id",
            "evaluation_seed": "(seed + 1) * stride + 50000 + index",
        },
    }
