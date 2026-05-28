from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .run_metadata import collect_run_metadata


def save_checkpoint(
    path: str | Path,
    agent: Any,
    global_step: int,
    episode_id: int,
    cfg: Any,
    extra_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": agent.state_dict(),
        "global_step": int(global_step),
        "episode_id": int(episode_id),
        "config": cfg,
        "metadata": collect_run_metadata(cfg, getattr(agent, "device", None)),
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, agent: Any, ride: Any | None = None) -> dict:
    state = torch.load(path, map_location=agent.device)
    agent.load_state_dict(state["agent"])
    if ride is not None and "ride" in state:
        ride.load_state_dict(state["ride"])
    return state
