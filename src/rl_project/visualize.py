from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch

from .config_utils import resolve_env_config
from .dqn_agent import DQNAgent
from .envs import make_env
from .preprocessing import preprocess_obs
from .trainer import resolve_device


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
        frames = []
        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)

        done = False
        truncated = False
        step = 0
        checkpoint_step = int(state.get("global_step", 0))
        while not (done or truncated):
            if max_steps is not None and step >= max_steps:
                break
            action = agent.act(obs, checkpoint_step, greedy=True)
            obs, _, done, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            step += 1
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No render frames were produced. Check MiniGrid render_mode support.")
    imageio.mimsave(output_path, frames, duration=1.0 / max(1, int(fps)))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a greedy policy rollout from a checkpoint to GIF.")
    parser.add_argument("checkpoint", help="Path to a checkpoint, e.g. outputs/.../checkpoints/step_200.pt")
    parser.add_argument("--out", default="visualizations/rollout.gif", help="Output GIF path.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed for the rendered rollout.")
    parser.add_argument("--fps", type=int, default=6, help="GIF frames per second.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional rollout step limit.")
    args = parser.parse_args()
    output_path = render_rollout(
        args.checkpoint,
        args.out,
        seed=args.seed,
        fps=args.fps,
        device_name=args.device,
        max_steps=args.max_steps,
    )
    print(f"Rollout written to: {output_path}")


if __name__ == "__main__":
    main()
