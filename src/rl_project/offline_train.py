from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from .config_utils import resolve_env_config
from .datasets import collect_fixed_replay, load_replay_rows
from .dqn_agent import DQNAgent
from .envs import make_env
from .logging_utils import RunLogger
from .nstep import NStepTransitionBuffer
from .replay import ReplayBuffer, Transition
from .run_context import resolve_output_dir
from .run_metadata import collect_run_metadata
from .trainer import apply_smoke_overrides, resolve_device


def _row_to_transition(row: dict) -> Transition:
    return Transition(
        obs=np.asarray(row["obs"], dtype=np.float32),
        action=int(row["action"]),
        reward_ext=float(row["reward_ext"]),
        next_obs=np.asarray(row["next_obs"], dtype=np.float32),
        done=bool(row["done"]),
        truncated=bool(row["truncated"]),
        env_seed=int(row["env_seed"]),
        episode_id=int(row["episode_id"]),
        timestep=int(row["timestep"]),
        picked_key=bool(row["picked_key"]),
        opened_door=bool(row["opened_door"]),
        entered_second_room=bool(row["entered_second_room"]),
        reached_goal=bool(row["reached_goal"]),
        timeout=bool(row["timeout"]),
        pickup_attempt=bool(row["pickup_attempt"]),
        toggle_attempt=bool(row["toggle_attempt"]),
        cell_position=tuple(row["cell_position"]),
        actual_n=int(row.get("actual_n", 1)),
    )


def _successful_episode_ids(rows: list[dict]) -> set[int]:
    return {int(row["episode_id"]) for row in rows if bool(row["reached_goal"])}


def _target_value(agent: DQNAgent, transition: Transition, gamma: float, device: torch.device) -> float:
    obs = torch.as_tensor(transition.next_obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if bool(agent.cfg.double_dqn):
            next_action = torch.argmax(agent.online(obs), dim=1)
            next_q = agent.target(obs).gather(1, next_action.unsqueeze(1)).squeeze(1)
        else:
            next_q = agent.target(obs).max(dim=1).values
        bootstrap = 0.0 if transition.done else float(gamma) ** int(transition.actual_n)
        return float(transition.reward_ext + bootstrap * float(next_q.item()))


def _diagnostic_rows(
    agent: DQNAgent,
    transitions: list[Transition],
    step: int,
    cfg: DictConfig,
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    gamma = float(cfg.agent.gamma)
    for transition in transitions:
        obs = torch.as_tensor(transition.obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_value = float(agent.online(obs)[0, int(transition.action)].item())
        target = _target_value(agent, transition, gamma, device)
        rows.append(
            {
                "update_step": int(step),
                "algorithm": str(cfg.algorithm.name),
                "episode_id": int(transition.episode_id),
                "prefix_timestep": int(transition.timestep),
                "action": int(transition.action),
                "n_step": int(cfg.algorithm.n_step),
                "actual_n": int(transition.actual_n),
                "q_value": q_value,
                "target": target,
                "bellman_error": target - q_value,
                "reward_ext": float(transition.reward_ext),
                "done": bool(transition.done),
                "truncated": bool(transition.truncated),
            }
        )
    return rows


def _write_diagnostics(run_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    frame.to_parquet(tables_dir / "offline_success_prefix_diagnostics.parquet", index=False)
    frame.to_csv(tables_dir / "offline_success_prefix_diagnostics.csv", index=False)


def run_offline_training(cfg: DictConfig) -> Path:
    OmegaConf.set_struct(cfg, False)
    apply_smoke_overrides(cfg)
    device = resolve_device(str(cfg.device))
    logger = RunLogger(resolve_output_dir(), cfg)
    data_path = cfg.replay.dataset_path
    if data_path is None:
        data_path = collect_fixed_replay(cfg, logger.run_dir / "fixed_replay")
    rows = load_replay_rows(data_path)
    successful_ids = _successful_episode_ids(rows)
    first = _row_to_transition(rows[0])
    buffer = ReplayBuffer(int(cfg.agent.replay_capacity), first.obs.shape, device)
    nstep = NStepTransitionBuffer(int(cfg.algorithm.n_step), float(cfg.agent.gamma))
    nstep_transitions: list[Transition] = []
    for row in rows:
        for ready in nstep.append(_row_to_transition(row)):
            buffer.add(ready)
            nstep_transitions.append(ready)
    success_prefix = [item for item in nstep_transitions if int(item.episode_id) in successful_ids]
    env = make_env(resolve_env_config(cfg))
    action_dim = int(env.action_space.n)
    env.close()
    agent = DQNAgent(first.obs.shape[0], action_dim, cfg.agent, device)
    requested_updates = int(cfg.replay.offline_updates)
    updates = min(requested_updates, 10) if cfg.smoke.enabled else requested_updates
    diagnostic_interval = max(1, int(cfg.replay.diagnostic_interval))
    diagnostic_table: list[dict] = []
    for step in range(updates):
        batch = buffer.sample(min(int(cfg.agent.batch_size), len(buffer)))
        update = agent.update(batch, float(cfg.agent.gamma), int(cfg.algorithm.n_step))
        logger.scalar("offline/loss", update.loss, step)
        logger.scalar("offline/q_mean", update.q_mean, step)
        if success_prefix and (step % diagnostic_interval == 0 or step == updates - 1):
            diagnostic_table.extend(_diagnostic_rows(agent, success_prefix, step, cfg, device))
    _write_diagnostics(logger.run_dir, diagnostic_table)
    torch.save(
        {
            "agent": agent.state_dict(),
            "dataset_path": str(data_path),
            "metadata": collect_run_metadata(cfg, device),
        },
        logger.ckpt_dir / "offline_final.pt",
    )
    logger.close()
    print(f"Offline run complete: {logger.run_dir}")
    return logger.run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_offline_training(cfg)


if __name__ == "__main__":
    main()
