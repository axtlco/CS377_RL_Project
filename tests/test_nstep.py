from __future__ import annotations

import numpy as np

from rl_project.nstep import NStepTransitionBuffer
from rl_project.replay import Transition


def tr(i: int, reward: float = 1.0, done: bool = False, truncated: bool = False) -> Transition:
    obs = np.asarray([i], dtype=np.float32)
    return Transition(
        obs=obs,
        action=0,
        reward_ext=reward,
        next_obs=obs + 1,
        done=done,
        truncated=truncated,
        env_seed=0,
        episode_id=0,
        timestep=i,
    )


def test_nstep_terminal_transition_truncates_return() -> None:
    buf = NStepTransitionBuffer(n_step=3, gamma=0.9)
    assert buf.append(tr(0, 1.0)) == []
    ready = buf.append(tr(1, 2.0, done=True))

    assert len(ready) == 2
    assert ready[0].reward_ext == 1.0 + 0.9 * 2.0
    assert ready[0].done
    assert ready[0].actual_n == 2
    assert ready[0].reward_ext_sequence == (1.0, 2.0)
    assert ready[1].reward_ext == 2.0
    assert ready[1].actual_n == 1
    assert ready[1].reward_ext_sequence == (2.0,)


def test_nstep_one_step_matches_plain_dqn_target_inputs() -> None:
    buf = NStepTransitionBuffer(n_step=1, gamma=0.99)
    ready = buf.append(tr(0, 3.5))

    assert len(ready) == 1
    assert ready[0].reward_ext == 3.5
    assert ready[0].actual_n == 1
    assert np.array_equal(ready[0].next_obs, np.asarray([1], dtype=np.float32))


def test_nstep_uses_actual_n_for_bootstrap_discount() -> None:
    buf = NStepTransitionBuffer(n_step=3, gamma=0.5)
    buf.append(tr(0, 1.0))
    buf.append(tr(1, 1.0))
    ready = buf.append(tr(2, 1.0))

    assert ready[0].reward_ext == 1.0 + 0.5 + 0.25
    assert ready[0].actual_n == 3


def test_nstep_accumulates_training_and_ride_rewards() -> None:
    buf = NStepTransitionBuffer(n_step=2, gamma=0.5)
    first = tr(0, 1.0)
    first.reward_train = 1.25
    first.reward_ride = 0.25
    second = tr(1, 2.0)
    second.reward_train = 2.5
    second.reward_ride = 0.5
    ready = buf.append(first)
    assert ready == []

    ready = buf.append(second)

    assert ready[0].reward_ext == 2.0
    assert ready[0].reward_train == 2.5
    assert ready[0].reward_ride == 0.5
    assert ready[0].reward_ext_sequence == (1.0, 2.0)
    assert len(ready[0].obs_sequence) == 2
    assert len(ready[0].next_obs_sequence) == 2
