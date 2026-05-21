from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .run_metadata import collect_run_metadata


def save_checkpoint(path: str | Path, agent: Any, global_step: int, episode_id: int, cfg: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
                "global_step": int(global_step),
                "episode_id": int(episode_id),
                "config": cfg,
                "metadata": collect_run_metadata(cfg, getattr(agent, "device", None)),
            },
        path,
    )


def load_checkpoint(path: str | Path, agent: Any) -> dict:
    state = torch.load(path, map_location=agent.device)
    agent.load_state_dict(state["agent"])
    return state
