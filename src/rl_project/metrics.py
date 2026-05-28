from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeTracker:
    run_id: str
    algorithm: str
    package: str
    env_id: str
    seed: int
    episode_id: int
    global_step: int
    episode_return_ext: float = 0.0
    episode_return_train: float = 0.0
    episode_return_ride: float = 0.0
    episode_length: int = 0
    success: bool = False
    timeout: bool = False
    picked_key: bool = False
    opened_door: bool = False
    entered_second_room: bool = False
    pickup_attempts: int = 0
    toggle_attempts: int = 0
    unique_cells_seen: set[tuple[int, int]] = field(default_factory=set)
    first_key_step: int | None = None
    first_door_step: int | None = None
    first_room_step: int | None = None
    first_success_step: int | None = None

    def update(
        self,
        reward_ext: float,
        diag: dict,
        reward_train: float | None = None,
        reward_ride: float = 0.0,
    ) -> None:
        self.episode_return_ext += float(reward_ext)
        self.episode_return_train += float(reward_ext if reward_train is None else reward_train)
        self.episode_return_ride += float(reward_ride)
        self.episode_length += 1
        self.success = self.success or bool(diag["reached_goal"])
        self.timeout = self.timeout or bool(diag["timeout"])
        self.picked_key = self.picked_key or bool(diag["picked_key"])
        self.opened_door = self.opened_door or bool(diag["opened_door"])
        self.entered_second_room = self.entered_second_room or bool(diag["entered_second_room"])
        self.pickup_attempts += int(diag["pickup_attempt"])
        self.toggle_attempts += int(diag["toggle_attempt"])
        self.unique_cells_seen.add(tuple(diag["cell_position"]))
        if diag["picked_key"] and self.first_key_step is None:
            self.first_key_step = self.episode_length
        if diag["opened_door"] and self.first_door_step is None:
            self.first_door_step = self.episode_length
        if diag["entered_second_room"] and self.first_room_step is None:
            self.first_room_step = self.episode_length
        if diag["reached_goal"] and self.first_success_step is None:
            self.first_success_step = self.episode_length

    def row(self) -> dict:
        return {
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "package": self.package,
            "env_id": self.env_id,
            "seed": self.seed,
            "global_step": self.global_step,
            "episode_id": self.episode_id,
            "episode_return_ext": self.episode_return_ext,
            "episode_return_train": self.episode_return_train,
            "episode_return_ride": self.episode_return_ride,
            "episode_length": self.episode_length,
            "success": self.success,
            "timeout": self.timeout,
            "picked_key": self.picked_key,
            "opened_door": self.opened_door,
            "entered_second_room": self.entered_second_room,
            "pickup_attempts": self.pickup_attempts,
            "toggle_attempts": self.toggle_attempts,
            "unique_cells": len(self.unique_cells_seen),
            "first_key_step": self.first_key_step,
            "first_door_step": self.first_door_step,
            "first_room_step": self.first_room_step,
            "first_success_step": self.first_success_step,
        }
