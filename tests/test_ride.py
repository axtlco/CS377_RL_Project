from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_project.ride import RIDEModule


def test_ride_intrinsic_reward_and_update_shapes() -> None:
    cfg = SimpleNamespace(
        beta=0.1,
        embedding_dim=8,
        hidden_dim=16,
        learning_rate=0.001,
        forward_loss_coef=1.0,
        inverse_loss_coef=1.0,
        reward_normalization="none",
        reward_clip_min=None,
        reward_clip_max=None,
        detach_embedding_targets=False,
        max_grad_norm=10.0,
    )
    ride = RIDEModule(obs_dim=4, action_dim=3, cfg=cfg, device=torch.device("cpu"))
    obs = torch.zeros(4).numpy()
    next_obs = torch.ones(4).numpy()

    reward = ride.intrinsic_reward(obs, next_obs, count_scale=0.5)

    assert reward.reward >= 0.0
    batch = {
        "obs": torch.zeros((5, 4), dtype=torch.float32),
        "next_obs": torch.ones((5, 4), dtype=torch.float32),
        "actions": torch.zeros(5, dtype=torch.long),
        "reward_ext": torch.zeros(5, dtype=torch.float32),
        "reward_ride": torch.ones(5, dtype=torch.float32),
        "ride_count_scale": torch.ones(5, dtype=torch.float32),
    }

    update = ride.update(batch)

    assert update.auxiliary_loss >= 0.0
    assert ride.update_count == 1
