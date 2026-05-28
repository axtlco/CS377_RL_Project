from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


DIAGNOSTIC_FIELDS = (
    "env_seed",
    "episode_id",
    "timestep",
    "picked_key",
    "opened_door",
    "entered_second_room",
    "reached_goal",
    "timeout",
    "pickup_attempt",
    "toggle_attempt",
    "cell_position",
)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward_ext: float
    next_obs: np.ndarray
    done: bool
    truncated: bool
    env_seed: int
    episode_id: int
    timestep: int
    picked_key: bool = False
    opened_door: bool = False
    entered_second_room: bool = False
    reached_goal: bool = False
    timeout: bool = False
    pickup_attempt: bool = False
    toggle_attempt: bool = False
    cell_position: tuple[int, int] = (0, 0)
    actual_n: int = 1
    reward_train: float | None = None
    reward_ride: float = 0.0
    ride_count_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.reward_train is None:
            self.reward_train = float(self.reward_ext)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["obs"] = self.obs.tolist()
        row["next_obs"] = self.next_obs.tolist()
        row["cell_position"] = list(self.cell_position)
        return row


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device | str = "cpu") -> None:
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device)
        self._data: list[Transition | None] = [None] * self.capacity
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        self._data[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Any]:
        if self._size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {self._size}")
        idxs = np.random.randint(0, self._size, size=int(batch_size))
        batch = [self._data[i] for i in idxs]
        assert all(item is not None for item in batch)
        items = [item for item in batch if item is not None]
        return {
            "obs": torch.as_tensor(np.stack([t.obs for t in items]), dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor([t.action for t in items], dtype=torch.long, device=self.device),
            "reward_ext": torch.as_tensor([t.reward_ext for t in items], dtype=torch.float32, device=self.device),
            "reward_train": torch.as_tensor([t.reward_train for t in items], dtype=torch.float32, device=self.device),
            "reward_ride": torch.as_tensor([t.reward_ride for t in items], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(np.stack([t.next_obs for t in items]), dtype=torch.float32, device=self.device),
            "done": torch.as_tensor([t.done for t in items], dtype=torch.bool, device=self.device),
            "truncated": torch.as_tensor([t.truncated for t in items], dtype=torch.bool, device=self.device),
            "actual_n": torch.as_tensor([t.actual_n for t in items], dtype=torch.long, device=self.device),
            "ride_count_scale": torch.as_tensor([t.ride_count_scale for t in items], dtype=torch.float32, device=self.device),
            "picked_key": torch.as_tensor([t.picked_key for t in items], dtype=torch.bool, device=self.device),
            "opened_door": torch.as_tensor([t.opened_door for t in items], dtype=torch.bool, device=self.device),
            "entered_second_room": torch.as_tensor([t.entered_second_room for t in items], dtype=torch.bool, device=self.device),
            "timeout": torch.as_tensor([t.timeout for t in items], dtype=torch.bool, device=self.device),
        }

    def rows(self) -> list[dict[str, Any]]:
        return [t.to_row() for t in self._data[: self._size] if t is not None]
