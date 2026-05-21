from __future__ import annotations

from typing import Any

import gymnasium as gym
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.wrappers import ImgObsWrapper


CUSTOM_DOORKEY_IDS = {
    "MiniGrid-DoorKey-12x12-v0": 12,
    "MiniGrid-DoorKey-16x16-v0": 16,
}


def register_custom_doorkey_envs() -> None:
    registry = gym.envs.registration.registry
    for env_id, size in CUSTOM_DOORKEY_IDS.items():
        if env_id in registry:
            continue
        gym.register(
            id=env_id,
            entry_point="minigrid.envs.doorkey:DoorKeyEnv",
            kwargs={"size": size},
        )


def make_env(env_cfg: Any, seed: int | None = None) -> gym.Env:
    register_custom_doorkey_envs()
    kwargs = {}
    if getattr(env_cfg, "max_steps", None) is not None:
        kwargs["max_steps"] = int(env_cfg.max_steps)
    env = gym.make(str(env_cfg.id), **kwargs)
    env = ImgObsWrapper(env)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    return env


def extract_diagnostics(env: gym.Env, action: int, reward: float, done: bool, truncated: bool) -> dict[str, Any]:
    base = env.unwrapped
    carrying = getattr(base, "carrying", None)
    agent_pos = tuple(int(v) for v in getattr(base, "agent_pos", (0, 0)))
    opened_door = False
    door_x = None
    grid = getattr(base, "grid", None)
    if grid is not None:
        for x in range(getattr(grid, "width", 0)):
            for y in range(getattr(grid, "height", 0)):
                obj = grid.get(x, y)
                if getattr(obj, "type", None) == "door":
                    opened_door = opened_door or bool(getattr(obj, "is_open", False))
                    door_x = x if door_x is None else door_x
    entered_second_room = bool(door_x is not None and agent_pos[0] > door_x)
    return {
        "picked_key": bool(getattr(carrying, "type", None) == "key"),
        "opened_door": opened_door,
        "entered_second_room": entered_second_room,
        "reached_goal": bool(done and reward > 0.0),
        "timeout": bool(truncated),
        "pickup_attempt": bool(action == 3),
        "toggle_attempt": bool(action == 5),
        "cell_position": agent_pos,
    }
