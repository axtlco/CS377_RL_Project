from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from PIL import Image, ImageDraw

from .config_utils import resolve_env_config
from .dqn_agent import DQNAgent
from .envs import make_env
from .preprocessing import preprocess_obs
from .ride import RIDEModule
from .trainer import resolve_device


ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def torch_load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def render_rollout(
    checkpoint_path: str | Path,
    output_path: str | Path,
    seed: int | None = None,
    fps: int = 6,
    device_name: str = "auto",
    max_steps: int | None = None,
    trace_path: str | Path | None = None,
    overlay: bool = True,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    state = torch_load_checkpoint(checkpoint_path, device)
    cfg = state["config"]

    env = make_env(resolve_env_config(cfg), seed=seed, render_mode="rgb_array")
    try:
        reset_seed = int(cfg.seed) if seed is None else int(seed)
        obs, _ = env.reset(seed=reset_seed)
        obs_vec = preprocess_obs(obs)
        agent = DQNAgent(obs_vec.shape[0], int(env.action_space.n), cfg.agent, device)
        agent.load_state_dict(state["agent"])
        ride = None
        ride_counts: dict[tuple[float, ...], int] = {}
        if bool(getattr(getattr(cfg, "ride", None), "enabled", False)) and "ride" in state:
            ride = RIDEModule(obs_vec.shape[0], int(env.action_space.n), cfg.ride, device)
            ride.load_state_dict(state["ride"])
            update_ride_count(ride_counts, obs_vec)
        frames = []
        trace_rows: list[dict[str, Any]] = []
        first_frame = env.render()
        if first_frame is not None:
            frames.append(
                annotate_frame(
                    first_frame,
                    step=0,
                    action=None,
                    reward=0.0,
                    reward_ride=None,
                    done=False,
                    truncated=False,
                    env=env,
                )
                if overlay
                else first_frame
            )

        done = False
        truncated = False
        step = 0
        checkpoint_step = int(state.get("global_step", 0))
        while not (done or truncated):
            if max_steps is not None and step >= max_steps:
                break
            obs_vec = preprocess_obs(obs)
            action = agent.act(obs, checkpoint_step, greedy=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            reward_ext = float(reward)
            ride_reward = None
            if ride is not None:
                next_obs_vec = preprocess_obs(next_obs)
                count_scale = ride_count_scale(cfg, ride_counts, next_obs_vec)
                ride_reward = ride.intrinsic_reward(obs_vec, next_obs_vec, count_scale=count_scale)
            trace_rows.append(
                trace_row(
                    env,
                    step + 1,
                    action,
                    reward_ext,
                    bool(done),
                    bool(truncated),
                    ride_reward=ride_reward,
                )
            )
            frame = env.render()
            if frame is not None:
                frames.append(
                    annotate_frame(
                        frame,
                        step + 1,
                        action,
                        reward_ext,
                        None if ride_reward is None else ride_reward.reward,
                        bool(done),
                        bool(truncated),
                        env,
                    )
                    if overlay
                    else frame
                )
            obs = next_obs
            step += 1
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No render frames were produced. Check MiniGrid render_mode support.")
    imageio.mimsave(output_path, frames, duration=1.0 / max(1, int(fps)))
    if trace_path is None:
        trace_path = output_path.with_suffix(".csv")
    write_trace(trace_path, trace_rows)
    return output_path


def trace_row(
    env,
    step: int,
    action: int,
    reward: float,
    done: bool,
    truncated: bool,
    ride_reward: Any | None = None,
) -> dict[str, Any]:
    base = env.unwrapped
    pos = tuple(int(value) for value in getattr(base, "agent_pos", (-1, -1)))
    direction = int(getattr(base, "agent_dir", -1))
    carrying = getattr(base, "carrying", None)
    reward_ride = 0.0 if ride_reward is None else float(ride_reward.reward)
    return {
        "step": int(step),
        "action": int(action),
        "action_name": ACTION_NAMES.get(int(action), str(action)),
        "reward": float(reward),
        "reward_ext": float(reward),
        "reward_ride": reward_ride,
        "reward_train": float(reward) + reward_ride,
        "ride_control_reward": 0.0 if ride_reward is None else float(ride_reward.control_reward),
        "ride_count_scale": 1.0 if ride_reward is None else float(ride_reward.count_scale),
        "done": bool(done),
        "truncated": bool(truncated),
        "agent_x": pos[0],
        "agent_y": pos[1],
        "agent_dir": direction,
        "carrying": getattr(carrying, "type", None) or "",
    }


def write_trace(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "action",
        "action_name",
        "reward",
        "reward_ext",
        "reward_ride",
        "reward_train",
        "ride_control_reward",
        "ride_count_scale",
        "done",
        "truncated",
        "agent_x",
        "agent_y",
        "agent_dir",
        "carrying",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_frame(
    frame,
    step: int,
    action: int | None,
    reward: float,
    reward_ride: float | None,
    done: bool,
    truncated: bool,
    env,
):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    action_text = "reset" if action is None else f"{action}:{ACTION_NAMES.get(int(action), action)}"
    base = env.unwrapped
    pos = tuple(int(value) for value in getattr(base, "agent_pos", (-1, -1)))
    direction = int(getattr(base, "agent_dir", -1))
    carrying = getattr(getattr(base, "carrying", None), "type", "") or "-"
    text = f"step={step} action={action_text} pos={pos} dir={direction} carry={carrying} r={reward:.3f}"
    if reward_ride is not None:
        text += f" ri={reward_ride:.3f}"
    if done:
        text += " done"
    if truncated:
        text += " truncated"
    left, top, right, bottom = draw.textbbox((0, 0), text)
    pad = 4
    draw.rectangle((0, 0, right + 2 * pad, bottom + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return image


def update_ride_count(counts: dict[tuple[float, ...], int], obs_vec) -> int:
    key = tuple(float(value) for value in obs_vec)
    count = counts.get(key, 0) + 1
    counts[key] = count
    return count


def ride_count_scale(cfg: Any, counts: dict[tuple[float, ...], int], obs_vec) -> float:
    mode = str(cfg.ride.count_scale)
    if mode == "none":
        return 1.0
    if mode != "episodic":
        raise ValueError(f"Unknown ride.count_scale={mode!r}")
    count = update_ride_count(counts, obs_vec)
    return 1.0 / math.sqrt(count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a greedy policy rollout from a checkpoint to GIF.")
    parser.add_argument("checkpoint", help="Path to a checkpoint, e.g. outputs/.../checkpoints/step_200.pt")
    parser.add_argument("--out", default="visualizations/rollout.gif", help="Output GIF path.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed for the rendered rollout.")
    parser.add_argument("--fps", type=int, default=6, help="GIF frames per second.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional rollout step limit.")
    parser.add_argument("--trace-out", default=None, help="Optional rollout trace CSV path. Defaults to output GIF stem.")
    parser.add_argument("--no-overlay", action="store_true", help="Disable step/action text overlay.")
    args = parser.parse_args()
    output_path = render_rollout(
        args.checkpoint,
        args.out,
        seed=args.seed,
        fps=args.fps,
        device_name=args.device,
        max_steps=args.max_steps,
        trace_path=args.trace_out,
        overlay=not args.no_overlay,
    )
    print(f"Rollout written to: {output_path}")


if __name__ == "__main__":
    main()
