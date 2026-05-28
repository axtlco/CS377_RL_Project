from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .networks import QNetwork
from .preprocessing import preprocess_obs_tensor


@dataclass
class DQNUpdate:
    loss: float
    q_mean: float
    target_mean: float


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg, device: torch.device) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.device = device
        self.online = QNetwork(obs_dim, action_dim, int(cfg.hidden_dim)).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(cfg.learning_rate))
        self.update_count = 0
        self.optimizer_step_count = 0
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self, step: int) -> float:
        frac = min(1.0, max(0.0, step / float(self.cfg.epsilon_decay_steps)))
        return float(self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    @torch.no_grad()
    def act(self, obs: np.ndarray, step: int, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon(step)
        if random.random() < eps:
            return random.randrange(self.action_dim)
        x = preprocess_obs_tensor(obs, self.device)
        return int(torch.argmax(self.online(x), dim=1).item())

    def update(self, batch: dict[str, torch.Tensor], gamma: float, n_step_default: int) -> DQNUpdate:
        q_values = self.online(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if bool(self.cfg.double_dqn):
                next_actions = torch.argmax(self.online(batch["next_obs"]), dim=1)
                next_q = self.target(batch["next_obs"]).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target(batch["next_obs"]).max(dim=1).values
            not_done = ~batch["done"]
            actual_n = batch.get("actual_n")
            if actual_n is None:
                actual_n = torch.full_like(batch["actions"], int(n_step_default))
            discounts = torch.pow(torch.full_like(batch["reward_ext"], float(gamma)), actual_n.float())
            rewards = batch.get("reward_train", batch["reward_ext"])
            targets = rewards + not_done.float() * discounts * next_q
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1
        self.optimizer_step_count += 1
        if self.update_count % int(self.cfg.target_update_interval) == 0:
            self.target.load_state_dict(self.online.state_dict())
        return DQNUpdate(float(loss.item()), float(q_values.mean().item()), float(targets.mean().item()))

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "optimizer_step_count": self.optimizer_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state.get("update_count", 0))
        self.optimizer_step_count = int(state.get("optimizer_step_count", 0))
