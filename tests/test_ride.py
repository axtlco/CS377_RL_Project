from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_project.ride import RIDEModule
from rl_project.trainer import Trainer


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
    batch_reward = ride.intrinsic_reward_batch(
        torch.zeros((2, 4), dtype=torch.float32),
        torch.ones((2, 4), dtype=torch.float32),
        torch.full((2,), 0.5, dtype=torch.float32),
    )

    assert reward.reward >= 0.0
    assert batch_reward.reward.shape == (2,)
    assert torch.all(batch_reward.reward >= 0.0)
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


def test_trainer_recomputes_ride_reward_train_from_current_model() -> None:
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
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.ride = RIDEModule(obs_dim=4, action_dim=3, cfg=cfg, device=trainer.device)
    batch = {
        "reward_ext": torch.full((1,), 999.0, dtype=torch.float32),
        "reward_train": torch.full((1,), 999.0, dtype=torch.float32),
        "reward_ride": torch.full((1,), 999.0, dtype=torch.float32),
        "nstep_obs": torch.zeros((1, 2, 4), dtype=torch.float32),
        "nstep_next_obs": torch.ones((1, 2, 4), dtype=torch.float32),
        "nstep_reward_ext": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "nstep_ride_count_scale": torch.ones((1, 2), dtype=torch.float32),
        "nstep_mask": torch.ones((1, 2), dtype=torch.bool),
    }

    updated = trainer._with_current_ride_rewards(batch, gamma=0.5)

    assert float(updated["reward_ext"].item()) == 2.0
    assert float(updated["reward_train"].item()) != 999.0
    assert torch.allclose(updated["reward_train"], updated["reward_ext"] + updated["reward_ride"])
