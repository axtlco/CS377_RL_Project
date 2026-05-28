from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_project.dqn_agent import DQNAgent


def test_timeout_transitions_keep_bootstrap_but_done_transitions_do_not() -> None:
    cfg = SimpleNamespace(
        hidden_dim=4,
        learning_rate=0.0,
        target_update_interval=100,
        double_dqn=False,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    agent = DQNAgent(obs_dim=2, action_dim=2, cfg=cfg, device=torch.device("cpu"))
    for param in agent.online.parameters():
        param.data.zero_()
    for param in agent.target.parameters():
        param.data.zero_()
    agent.target.net[-1].bias.data.fill_(2.0)
    batch = {
        "obs": torch.zeros((2, 2), dtype=torch.float32),
        "actions": torch.zeros(2, dtype=torch.long),
        "reward_ext": torch.ones(2, dtype=torch.float32),
        "next_obs": torch.zeros((2, 2), dtype=torch.float32),
        "done": torch.tensor([False, True], dtype=torch.bool),
        "truncated": torch.tensor([True, False], dtype=torch.bool),
        "actual_n": torch.ones(2, dtype=torch.long),
    }

    update = agent.update(batch, gamma=0.5, n_step_default=1)

    assert update.target_mean == 1.5


def test_dqn_update_uses_training_reward_when_present() -> None:
    cfg = SimpleNamespace(
        hidden_dim=4,
        learning_rate=0.0,
        target_update_interval=100,
        double_dqn=False,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    agent = DQNAgent(obs_dim=2, action_dim=2, cfg=cfg, device=torch.device("cpu"))
    for param in agent.online.parameters():
        param.data.zero_()
    for param in agent.target.parameters():
        param.data.zero_()
    batch = {
        "obs": torch.zeros((1, 2), dtype=torch.float32),
        "actions": torch.zeros(1, dtype=torch.long),
        "reward_ext": torch.ones(1, dtype=torch.float32),
        "reward_train": torch.full((1,), 3.0, dtype=torch.float32),
        "next_obs": torch.zeros((1, 2), dtype=torch.float32),
        "done": torch.ones(1, dtype=torch.bool),
        "actual_n": torch.ones(1, dtype=torch.long),
    }

    update = agent.update(batch, gamma=0.99, n_step_default=1)

    assert update.target_mean == 3.0
