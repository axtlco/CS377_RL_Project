from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config_utils import env_key, resolve_env_config
from .envs import extract_diagnostics, make_env


@torch.no_grad()
def evaluate_agent(
    agent,
    cfg: Any,
    eval_seeds: list[int],
    checkpoint_step: int,
    run_id: str,
) -> list[dict]:
    weight_snapshot = {k: v.detach().clone() for k, v in agent.online.state_dict().items()}
    update_count = agent.optimizer_step_count
    rows: list[dict] = []
    env = make_env(resolve_env_config(cfg))
    try:
        for eval_seed in eval_seeds:
            obs, _ = env.reset(seed=int(eval_seed))
            done = False
            truncated = False
            ep_return = 0.0
            ep_len = 0
            aggregate = {
                "picked_key": False,
                "opened_door": False,
                "entered_second_room": False,
                "success": False,
            }
            while not (done or truncated):
                action = agent.act(obs, checkpoint_step, greedy=True)
                next_obs, reward, done, truncated, _ = env.step(action)
                diag = extract_diagnostics(env, action, float(reward), bool(done), bool(truncated))
                aggregate["picked_key"] = aggregate["picked_key"] or diag["picked_key"]
                aggregate["opened_door"] = aggregate["opened_door"] or diag["opened_door"]
                aggregate["entered_second_room"] = aggregate["entered_second_room"] or diag["entered_second_room"]
                aggregate["success"] = aggregate["success"] or diag["reached_goal"]
                ep_return += float(reward)
                ep_len += 1
                obs = next_obs
            rows.append(
                {
                    "run_id": run_id,
                    "algorithm": cfg.algorithm.name,
                    "package": cfg.package.name,
                    "env_id": env_key(cfg),
                    "seed": int(cfg.seed),
                    "checkpoint_step": int(checkpoint_step),
                    "eval_seed": int(eval_seed),
                    "success": aggregate["success"],
                    "return_ext": ep_return,
                    "episode_length": ep_len,
                    "picked_key": aggregate["picked_key"],
                    "opened_door": aggregate["opened_door"],
                    "entered_second_room": aggregate["entered_second_room"],
                }
            )
    finally:
        env.close()
    for key, value in agent.online.state_dict().items():
        if not torch.equal(value, weight_snapshot[key]):
            raise RuntimeError("Evaluation changed model weights")
    if agent.optimizer_step_count != update_count:
        raise RuntimeError("Evaluation changed optimizer step count")
    return rows


def summarize_eval(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"eval/success_rate": 0.0, "eval/return_ext_mean": 0.0}
    return {
        "eval/success_rate": float(np.mean([row["success"] for row in rows])),
        "eval/return_ext_mean": float(np.mean([row["return_ext"] for row in rows])),
    }
