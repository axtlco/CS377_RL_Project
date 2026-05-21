from __future__ import annotations

import json
import random
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import pandas as pd

from .config_utils import env_key, resolve_env_config
from .envs import extract_diagnostics, make_env
from .preprocessing import preprocess_obs
from .replay import Transition
from .seeding import SeedStream, set_global_seeds


LEFT = 0
RIGHT = 1
FORWARD = 2
PICKUP = 3
TOGGLE = 5
DIR_TO_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


@dataclass
class EpisodeRows:
    rows: list[dict[str, Any]]
    success: bool
    collector_policy: str


def _cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg.replay, name, default) if hasattr(cfg, "replay") else default


def _target_success_episodes(cfg: Any) -> int:
    explicit = _cfg_value(cfg, "target_success_episodes")
    if explicit is not None:
        return int(explicit)
    ratio = _cfg_value(cfg, "target_success_ratio")
    if ratio is None:
        return 0
    ratio = float(ratio)
    if ratio <= 0.0:
        return 0
    raw = int(cfg.replay.episode_count) * ratio
    if raw < 1.0:
        raise ValueError(
            "target_success_ratio is nonzero but episode_count * target_success_ratio < 1. "
            "Increase replay.episode_count or set replay.target_success_episodes=1."
        )
    return int(round(raw))


def _object_positions(env: gym.Env) -> dict[str, tuple[int, int]]:
    base = env.unwrapped
    grid = getattr(base, "grid", None)
    if grid is None:
        raise RuntimeError("DoorKey scripted collector requires access to env.unwrapped.grid")
    positions: dict[str, tuple[int, int]] = {}
    for x in range(getattr(grid, "width", 0)):
        for y in range(getattr(grid, "height", 0)):
            obj = grid.get(x, y)
            obj_type = getattr(obj, "type", None)
            if obj_type in {"key", "door", "goal"} and obj_type not in positions:
                positions[str(obj_type)] = (x, y)
    missing = {"key", "door", "goal"} - set(positions)
    if missing:
        raise RuntimeError(f"Could not find DoorKey objects: {sorted(missing)}")
    return positions


def _cell_type(env: gym.Env, pos: tuple[int, int]) -> str | None:
    obj = env.unwrapped.grid.get(*pos)
    return getattr(obj, "type", None)


def _is_walkable(env: gym.Env, pos: tuple[int, int], door_open: bool) -> bool:
    x, y = pos
    grid = env.unwrapped.grid
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    obj_type = _cell_type(env, pos)
    if obj_type is None or obj_type == "goal":
        return True
    if obj_type == "door":
        obj = grid.get(x, y)
        return bool(door_open or getattr(obj, "is_open", False))
    return False


def _neighbors(env: gym.Env, pos: tuple[int, int], door_open: bool) -> Iterable[tuple[int, int]]:
    x, y = pos
    for dx, dy in DIR_TO_VEC.values():
        nxt = (x + dx, y + dy)
        if _is_walkable(env, nxt, door_open):
            yield nxt


def _shortest_path(
    env: gym.Env,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    door_open: bool,
) -> list[tuple[int, int]] | None:
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        pos = queue.popleft()
        if pos in goals:
            path: list[tuple[int, int]] = []
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return list(reversed(path))
        for nxt in _neighbors(env, pos, door_open):
            if nxt not in parent:
                parent[nxt] = pos
                queue.append(nxt)
    return None


def _dir_from_delta(delta: tuple[int, int]) -> int:
    for direction, vec in DIR_TO_VEC.items():
        if vec == delta:
            return direction
    raise ValueError(f"Invalid movement delta: {delta}")


def _turn_actions(current_dir: int, target_dir: int) -> tuple[list[int], int]:
    right_turns = (target_dir - current_dir) % 4
    left_turns = (current_dir - target_dir) % 4
    if left_turns < right_turns:
        return [LEFT] * left_turns, target_dir
    return [RIGHT] * right_turns, target_dir


def _path_actions(path: list[tuple[int, int]], start_dir: int) -> tuple[list[int], int]:
    actions: list[int] = []
    direction = int(start_dir)
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        target_dir = _dir_from_delta(delta)
        turns, direction = _turn_actions(direction, target_dir)
        actions.extend(turns)
        actions.append(FORWARD)
    return actions, direction


def _adjacent_goals(env: gym.Env, target: tuple[int, int], door_open: bool) -> dict[tuple[int, int], int]:
    goals: dict[tuple[int, int], int] = {}
    tx, ty = target
    for direction, vec in DIR_TO_VEC.items():
        stand = (tx - vec[0], ty - vec[1])
        if _is_walkable(env, stand, door_open):
            goals[stand] = direction
    return goals


def _move_to_adjacent_and_face(
    env: gym.Env,
    target: tuple[int, int],
    start_pos: tuple[int, int],
    start_dir: int,
    door_open: bool,
) -> tuple[list[int], tuple[int, int], int]:
    goals = _adjacent_goals(env, target, door_open)
    path = _shortest_path(env, start_pos, set(goals), door_open)
    if path is None:
        raise RuntimeError(f"No path to adjacent cell for target {target}")
    actions, direction = _path_actions(path, start_dir)
    face_dir = goals[path[-1]]
    turns, direction = _turn_actions(direction, face_dir)
    actions.extend(turns)
    return actions, path[-1], direction


def scripted_doorkey_actions(env: gym.Env) -> list[int]:
    positions = _object_positions(env)
    pos = tuple(int(v) for v in env.unwrapped.agent_pos)
    direction = int(env.unwrapped.agent_dir)
    actions: list[int] = []

    step_actions, pos, direction = _move_to_adjacent_and_face(env, positions["key"], pos, direction, door_open=False)
    actions.extend(step_actions)
    actions.append(PICKUP)

    step_actions, pos, direction = _move_to_adjacent_and_face(env, positions["door"], pos, direction, door_open=False)
    actions.extend(step_actions)
    actions.append(TOGGLE)

    path = _shortest_path(env, pos, {positions["goal"]}, door_open=True)
    if path is None:
        raise RuntimeError("No path from opened door to goal")
    step_actions, direction = _path_actions(path, direction)
    actions.extend(step_actions)
    return actions


def _episode_from_actions(
    env: gym.Env,
    actions: list[int],
    episode_id: int,
    env_seed: int,
    collector_policy: str,
) -> EpisodeRows:
    obs, _ = env.reset(seed=env_seed)
    env.action_space.seed(env_seed)
    rows: list[dict[str, Any]] = []
    success = False
    done = False
    truncated = False
    timestep = 0
    for action in actions:
        if done or truncated:
            break
        next_obs, reward, done, truncated, _ = env.step(int(action))
        diag = extract_diagnostics(env, int(action), float(reward), bool(done), bool(truncated))
        success = success or bool(diag["reached_goal"])
        rows.append(
            Transition(
                obs=preprocess_obs(obs),
                action=int(action),
                reward_ext=float(reward),
                next_obs=preprocess_obs(next_obs),
                done=bool(done),
                truncated=bool(truncated),
                env_seed=int(env_seed),
                episode_id=int(episode_id),
                timestep=int(timestep),
                **diag,
            ).to_row()
        )
        obs = next_obs
        timestep += 1
    return EpisodeRows(rows=rows, success=success, collector_policy=collector_policy)


def _collect_policy_episode(
    env: gym.Env,
    policy: str,
    episode_id: int,
    env_seed: int,
    epsilon: float,
) -> EpisodeRows:
    if policy == "scripted_success":
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        return _episode_from_actions(env, scripted_doorkey_actions(env), episode_id, env_seed, policy)

    obs, _ = env.reset(seed=env_seed)
    env.action_space.seed(env_seed)
    scripted_actions: list[int] = []
    if policy == "epsilon_random":
        scripted_actions = scripted_doorkey_actions(env)
    rows: list[dict[str, Any]] = []
    success = False
    done = False
    truncated = False
    timestep = 0
    while not (done or truncated):
        if policy == "epsilon_random" and timestep < len(scripted_actions) and random.random() >= epsilon:
            action = int(scripted_actions[timestep])
        else:
            action = int(env.action_space.sample())
        next_obs, reward, done, truncated, _ = env.step(action)
        diag = extract_diagnostics(env, action, float(reward), bool(done), bool(truncated))
        success = success or bool(diag["reached_goal"])
        rows.append(
            Transition(
                obs=preprocess_obs(obs),
                action=action,
                reward_ext=float(reward),
                next_obs=preprocess_obs(next_obs),
                done=bool(done),
                truncated=bool(truncated),
                env_seed=int(env_seed),
                episode_id=int(episode_id),
                timestep=int(timestep),
                **diag,
            ).to_row()
        )
        obs = next_obs
        timestep += 1
    return EpisodeRows(rows=rows, success=success, collector_policy=policy)


def collect_fixed_replay(cfg: Any, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seeds(int(cfg.seed))
    selected_env = env_key(cfg)
    episode_count = int(cfg.replay.episode_count)
    target_successes = _target_success_episodes(cfg)
    policy = str(cfg.replay.collector_policy)
    epsilon = float(_cfg_value(cfg, "epsilon", 1.0))
    max_attempts = int(_cfg_value(cfg, "max_collection_attempts", max(episode_count * 20, 100)))
    if target_successes > episode_count:
        raise ValueError("target_success_episodes cannot exceed replay.episode_count")
    if policy not in {"random", "epsilon_random", "scripted_success", "mixed"}:
        raise ValueError(f"Unknown replay.collector_policy={policy!r}")

    env = make_env(resolve_env_config(cfg), int(cfg.seed))
    stream = SeedStream(int(cfg.seed))
    rows: list[dict[str, Any]] = []
    episode_results: list[EpisodeRows] = []
    attempt = 0
    try:
        if policy in {"random", "epsilon_random", "scripted_success"}:
            for episode_id in range(episode_count):
                result = _collect_policy_episode(env, policy, episode_id, stream.env_seed(attempt), epsilon)
                episode_results.append(result)
                attempt += 1
        else:
            episode_id = 0
            while sum(item.success for item in episode_results) < target_successes:
                if attempt >= max_attempts:
                    raise RuntimeError("Exceeded max_collection_attempts while collecting successful scripted episodes")
                result = _collect_policy_episode(
                    env, "scripted_success", episode_id, stream.env_seed(attempt), epsilon
                )
                attempt += 1
                if result.success:
                    episode_results.append(result)
                    episode_id += 1
            filler_policy = "epsilon_random" if epsilon < 1.0 else "random"
            while len(episode_results) < episode_count:
                if attempt >= max_attempts:
                    raise RuntimeError("Exceeded max_collection_attempts while collecting fixed replay filler episodes")
                result = _collect_policy_episode(env, filler_policy, episode_id, stream.env_seed(attempt), epsilon)
                attempt += 1
                if result.success and sum(item.success for item in episode_results) >= target_successes:
                    continue
                episode_results.append(result)
                episode_id += 1
    finally:
        env.close()

    for episode_id, result in enumerate(episode_results):
        for row in result.rows:
            row["episode_id"] = episode_id
            rows.append(row)
    successful = sum(result.success for result in episode_results)
    collector_counts = Counter(result.collector_policy for result in episode_results)
    dataset_id = f"{selected_env}_seed{cfg.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    data_path = output_dir / "transitions.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    metadata = {
        "dataset_id": dataset_id,
        "env_id": selected_env,
        "package": cfg.package.name,
        "dataset_seed": int(cfg.seed),
        "target_success_ratio": _cfg_value(cfg, "target_success_ratio"),
        "target_success_episodes": target_successes,
        "success_ratio": float(successful / max(1, episode_count)),
        "episode_count": episode_count,
        "transition_count": len(rows),
        "successful_episode_count": int(successful),
        "collector_policy": policy,
        "collector_episode_counts": dict(collector_counts),
        "collection_attempts": attempt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": int(cfg.schema_version),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_path


def collect_random_replay(cfg: Any, output_dir: str | Path) -> Path:
    return collect_fixed_replay(cfg, output_dir)


def load_replay_rows(path: str | Path) -> list[dict]:
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")
