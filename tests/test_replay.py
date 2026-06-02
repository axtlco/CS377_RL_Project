from __future__ import annotations

import numpy as np
import torch

from rl_project.replay import ReplayBuffer, Transition


def make_transition(i: int, done: bool = False, truncated: bool = False) -> Transition:
    obs = np.full((4,), i, dtype=np.float32)
    return Transition(
        obs=obs,
        action=i % 3,
        reward_ext=float(i),
        next_obs=obs + 1,
        done=done,
        truncated=truncated,
        env_seed=123,
        episode_id=1,
        timestep=i,
        picked_key=True,
        opened_door=i % 2 == 0,
        entered_second_room=False,
        timeout=truncated,
        cell_position=(i, i + 1),
        reward_train=float(i) + 0.25,
        reward_ride=0.25,
        ride_count_scale=0.5,
    )


def test_replay_sampling_shapes_dtypes_and_flags() -> None:
    buffer = ReplayBuffer(capacity=10, obs_shape=(4,), device="cpu")
    for i in range(6):
        buffer.add(make_transition(i, done=i == 5, truncated=i == 4))

    batch = buffer.sample(4)

    assert batch["obs"].shape == (4, 4)
    assert batch["next_obs"].dtype == torch.float32
    assert batch["actions"].dtype == torch.long
    assert batch["reward_train"].shape == (4,)
    assert batch["reward_ride"].dtype == torch.float32
    assert batch["ride_count_scale"].dtype == torch.float32
    assert batch["nstep_obs"].shape == (4, 1, 4)
    assert batch["nstep_next_obs"].shape == (4, 1, 4)
    assert batch["nstep_reward_ext"].shape == (4, 1)
    assert batch["nstep_ride_count_scale"].shape == (4, 1)
    assert batch["nstep_mask"].all()
    assert batch["done"].dtype == torch.bool
    assert batch["truncated"].dtype == torch.bool
    assert batch["picked_key"].all()
    assert "opened_door" in batch
    assert "entered_second_room" in batch
