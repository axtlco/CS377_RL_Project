from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RIDEIntrinsicReward:
    reward: float
    control_reward: float
    count_scale: float
    normalized_reward: float


@dataclass
class RIDEUpdate:
    auxiliary_loss: float
    forward_loss: float
    inverse_loss: float
    intrinsic_reward_mean: float
    control_reward_mean: float
    count_scale_mean: float


class RIDEStateEmbeddingNet(nn.Module):
    def __init__(self, obs_dim: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RIDEForwardDynamicsNet(nn.Module):
    def __init__(self, embedding_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, state_embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        return self.net(torch.cat((state_embedding, action_one_hot), dim=-1))


class RIDEInverseDynamicsNet(nn.Module):
    def __init__(self, embedding_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state_embedding: torch.Tensor, next_state_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state_embedding, next_state_embedding), dim=-1))


class RIDEModule:
    def __init__(self, obs_dim: int, action_dim: int, cfg: Any, device: torch.device) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.device = device
        self.embedding = RIDEStateEmbeddingNet(
            self.obs_dim,
            int(cfg.embedding_dim),
            int(cfg.hidden_dim),
        ).to(device)
        self.forward_dynamics = RIDEForwardDynamicsNet(
            int(cfg.embedding_dim),
            self.action_dim,
            int(cfg.hidden_dim),
        ).to(device)
        self.inverse_dynamics = RIDEInverseDynamicsNet(
            int(cfg.embedding_dim),
            self.action_dim,
            int(cfg.hidden_dim),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.learning_rate))
        self.update_count = 0
        self._reward_square_mean = 1.0
        self._normalizer_initialized = False

    def parameters(self):
        yield from self.embedding.parameters()
        yield from self.forward_dynamics.parameters()
        yield from self.inverse_dynamics.parameters()

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    @torch.no_grad()
    def intrinsic_reward(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        count_scale: float = 1.0,
    ) -> RIDEIntrinsicReward:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_embedding = self.embedding(obs_tensor)
        next_state_embedding = self.embedding(next_obs_tensor)
        control_reward = torch.norm(next_state_embedding - state_embedding, dim=-1, p=2)
        scale = torch.as_tensor([float(count_scale)], dtype=torch.float32, device=self.device)
        raw_reward = control_reward * scale
        normalized_reward = self._normalize_reward(raw_reward)
        clipped_reward = self._clip_reward(normalized_reward)
        reward = float(clipped_reward.item() * float(self.cfg.beta))
        return RIDEIntrinsicReward(
            reward=reward,
            control_reward=float(control_reward.item()),
            count_scale=float(count_scale),
            normalized_reward=float(normalized_reward.item()),
        )

    def update(self, batch: dict[str, torch.Tensor]) -> RIDEUpdate:
        state_embedding = self.embedding(batch["obs"])
        next_state_embedding = self.embedding(batch["next_obs"])
        forward_target = next_state_embedding.detach() if bool(self.cfg.detach_embedding_targets) else next_state_embedding
        pred_next_state_embedding = self.forward_dynamics(state_embedding, batch["actions"])
        pred_actions = self.inverse_dynamics(state_embedding, next_state_embedding)

        forward_loss_raw = torch.norm(pred_next_state_embedding - forward_target, dim=-1, p=2).mean()
        inverse_loss_raw = F.cross_entropy(pred_actions, batch["actions"])
        forward_loss = float(self.cfg.forward_loss_coef) * forward_loss_raw
        inverse_loss = float(self.cfg.inverse_loss_coef) * inverse_loss_raw
        auxiliary_loss = forward_loss + inverse_loss

        self.optimizer.zero_grad(set_to_none=True)
        auxiliary_loss.backward()
        nn.utils.clip_grad_norm_(list(self.parameters()), float(self.cfg.max_grad_norm))
        self.optimizer.step()
        self.update_count += 1

        reward_ride = batch.get("reward_ride")
        if reward_ride is None:
            reward_ride = torch.zeros_like(batch["reward_ext"])
        count_scale = batch.get("ride_count_scale")
        if count_scale is None:
            count_scale = torch.ones_like(batch["reward_ext"])
        control_reward = torch.norm(next_state_embedding.detach() - state_embedding.detach(), dim=-1, p=2)
        return RIDEUpdate(
            auxiliary_loss=float(auxiliary_loss.item()),
            forward_loss=float(forward_loss.item()),
            inverse_loss=float(inverse_loss.item()),
            intrinsic_reward_mean=float(reward_ride.mean().item()),
            control_reward_mean=float(control_reward.mean().item()),
            count_scale_mean=float(count_scale.float().mean().item()),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.state_dict(),
            "forward_dynamics": self.forward_dynamics.state_dict(),
            "inverse_dynamics": self.inverse_dynamics.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "reward_square_mean": self._reward_square_mean,
            "normalizer_initialized": self._normalizer_initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.embedding.load_state_dict(state["embedding"])
        self.forward_dynamics.load_state_dict(state["forward_dynamics"])
        self.inverse_dynamics.load_state_dict(state["inverse_dynamics"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state.get("update_count", 0))
        self._reward_square_mean = float(state.get("reward_square_mean", 1.0))
        self._normalizer_initialized = bool(state.get("normalizer_initialized", False))

    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.reward_normalization)
        if mode == "none":
            return reward
        if mode != "ema_std":
            raise ValueError(f"Unknown ride.reward_normalization={mode!r}")
        square_mean = float(torch.mean(reward.detach() ** 2).item())
        if not self._normalizer_initialized:
            self._reward_square_mean = max(square_mean, 1e-8)
            self._normalizer_initialized = True
        else:
            self._reward_square_mean = 0.99 * self._reward_square_mean + 0.01 * square_mean
        return reward / (self._reward_square_mean**0.5 + 1e-8)

    def _clip_reward(self, reward: torch.Tensor) -> torch.Tensor:
        clip_min = self.cfg.reward_clip_min
        clip_max = self.cfg.reward_clip_max
        if clip_min is None and clip_max is None:
            return reward
        min_value = None if clip_min is None else float(clip_min)
        max_value = None if clip_max is None else float(clip_max)
        return torch.clamp(reward, min=min_value, max=max_value)
