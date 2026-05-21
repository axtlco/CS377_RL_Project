from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from .checkpointing import save_checkpoint
from .config_utils import env_key, resolve_env_config
from .dqn_agent import DQNAgent
from .envs import extract_diagnostics, make_env
from .evaluate import evaluate_agent, summarize_eval
from .logging_utils import RunLogger
from .metrics import EpisodeTracker
from .nstep import NStepTransitionBuffer
from .preprocessing import preprocess_obs
from .replay import ReplayBuffer, Transition
from .run_context import resolve_output_dir
from .seeding import SeedStream, fixed_eval_seeds, set_global_seeds


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def env_value(mapping: Any, key: str, default: Any | None = None) -> Any:
    if key in mapping:
        return mapping[key]
    if "default" in mapping:
        return mapping["default"]
    return default


def apply_smoke_overrides(cfg: Any) -> None:
    if not bool(cfg.smoke.enabled):
        return
    selected_env = env_key(cfg)
    cfg.package.training_steps[selected_env] = int(cfg.smoke.training_steps)
    cfg.package.eval_interval[selected_env] = int(cfg.smoke.eval_interval)
    cfg.package.eval_episodes[selected_env] = int(cfg.smoke.eval_episodes)
    cfg.agent.learning_starts = int(cfg.smoke.learning_starts)
    cfg.agent.batch_size = int(cfg.smoke.batch_size)
    cfg.agent.replay_capacity = int(cfg.smoke.replay_capacity)


class Trainer:
    def __init__(self, cfg: Any) -> None:
        OmegaConf.set_struct(cfg, False)
        apply_smoke_overrides(cfg)
        self.cfg = cfg
        self.device = resolve_device(str(cfg.device))
        set_global_seeds(int(cfg.seed))
        self.env_cfg = resolve_env_config(cfg)
        self.run_id = f"{cfg.algorithm.name}_{env_key(cfg)}_seed{cfg.seed}"
        self.logger = RunLogger(resolve_output_dir(), cfg)
        self.env = make_env(self.env_cfg, seed=int(cfg.seed))
        obs, _ = self.env.reset(seed=int(cfg.seed))
        obs_vec = preprocess_obs(obs)
        self.obs_dim = int(obs_vec.shape[0])
        self.action_dim = int(self.env.action_space.n)
        self.agent = DQNAgent(self.obs_dim, self.action_dim, cfg.agent, self.device)
        self.replay = ReplayBuffer(int(cfg.agent.replay_capacity), obs_vec.shape, self.device)
        self.nstep = NStepTransitionBuffer(int(cfg.algorithm.n_step), float(cfg.agent.gamma))
        self.seed_stream = SeedStream(int(cfg.seed))

    def train(self) -> Path:
        cfg = self.cfg
        selected_env = env_key(cfg)
        training_steps = int(env_value(cfg.package.training_steps, selected_env))
        eval_interval = int(env_value(cfg.package.eval_interval, selected_env))
        eval_episodes = int(env_value(cfg.package.eval_episodes, selected_env, cfg.package.eval_episodes.default))
        eval_seeds = fixed_eval_seeds(int(cfg.seed), eval_episodes)
        progress = tqdm(total=training_steps, disable=not bool(cfg.logging.progress), desc=self.run_id)

        global_step = 0
        episode_id = 0
        env_seed = self.seed_stream.env_seed(episode_id)
        obs, _ = self.env.reset(seed=env_seed)
        tracker = self._new_tracker(episode_id, global_step)

        try:
            while global_step < training_steps:
                action = self.agent.act(obs, global_step)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                diag = extract_diagnostics(self.env, action, float(reward), bool(done), bool(truncated))
                transition = Transition(
                    obs=preprocess_obs(obs),
                    action=int(action),
                    reward_ext=float(reward),
                    next_obs=preprocess_obs(next_obs),
                    done=bool(done),
                    truncated=bool(truncated),
                    env_seed=int(env_seed),
                    episode_id=int(episode_id),
                    timestep=int(tracker.episode_length),
                    **diag,
                )
                for ready in self.nstep.append(transition):
                    self.replay.add(ready)
                tracker.update(float(reward), diag)
                obs = next_obs
                global_step += 1
                progress.update(1)

                if self._should_update(global_step):
                    for _ in range(int(cfg.agent.update_to_data_ratio)):
                        batch = self.replay.sample(int(cfg.agent.batch_size))
                        update = self.agent.update(batch, float(cfg.agent.gamma), int(cfg.algorithm.n_step))
                        self.logger.scalar("train/loss", update.loss, global_step)
                        self.logger.scalar("train/q_mean", update.q_mean, global_step)
                        self.logger.scalar("train/target_mean", update.target_mean, global_step)

                if done or truncated:
                    tracker.global_step = global_step
                    self.logger.episode(tracker.row())
                    self.logger.scalar("episode/return_ext", tracker.episode_return_ext, global_step)
                    self.logger.scalar("episode/success", float(tracker.success), global_step)
                    episode_id += 1
                    env_seed = self.seed_stream.env_seed(episode_id)
                    obs, _ = self.env.reset(seed=env_seed)
                    tracker = self._new_tracker(episode_id, global_step)

                if global_step % eval_interval == 0 or global_step == training_steps:
                    self._run_eval(eval_seeds, global_step)

                if global_step % int(cfg.logging.save_interval) == 0 or global_step == training_steps:
                    ckpt = self.logger.ckpt_dir / f"step_{global_step}.pt"
                    save_checkpoint(ckpt, self.agent, global_step, episode_id, cfg)
        finally:
            progress.close()
            self.env.close()
            self.logger.close()
        return self.logger.run_dir

    def _should_update(self, global_step: int) -> bool:
        cfg = self.cfg.agent
        return (
            global_step >= int(cfg.learning_starts)
            and len(self.replay) >= int(cfg.batch_size)
            and global_step % int(cfg.train_interval) == 0
        )

    def _run_eval(self, eval_seeds: list[int], global_step: int) -> None:
        replay_size = len(self.replay)
        rows = evaluate_agent(self.agent, self.cfg, eval_seeds, global_step, self.run_id)
        if len(self.replay) != replay_size:
            raise RuntimeError("Evaluation changed replay buffer size")
        self.logger.eval(rows)
        for name, value in summarize_eval(rows).items():
            self.logger.scalar(name, value, global_step)

    def _new_tracker(self, episode_id: int, global_step: int) -> EpisodeTracker:
        return EpisodeTracker(
            run_id=self.run_id,
            algorithm=str(self.cfg.algorithm.name),
            package=str(self.cfg.package.name),
            env_id=env_key(self.cfg),
            seed=int(self.cfg.seed),
            episode_id=int(episode_id),
            global_step=int(global_step),
        )


def run_training(cfg: Any) -> Path:
    return Trainer(cfg).train()
