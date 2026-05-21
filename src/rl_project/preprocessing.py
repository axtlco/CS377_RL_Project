from __future__ import annotations

import numpy as np
import torch


OBJECT_SCALE = 10.0
COLOR_SCALE = 5.0
STATE_SCALE = 3.0


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected MiniGrid symbolic image with shape HxWx3, got {arr.shape}")
    scaled = arr.copy()
    scaled[..., 0] /= OBJECT_SCALE
    scaled[..., 1] /= COLOR_SCALE
    scaled[..., 2] /= STATE_SCALE
    return scaled.reshape(-1).astype(np.float32)


def preprocess_obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(preprocess_obs(obs), dtype=torch.float32, device=device).unsqueeze(0)
