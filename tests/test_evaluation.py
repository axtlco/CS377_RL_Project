from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from rl_project.config_utils import resolve_env_config
from rl_project.dqn_agent import DQNAgent
from rl_project.envs import make_env
from rl_project.evaluate import evaluate_agent
from rl_project.preprocessing import preprocess_obs
from rl_project.trainer import resolve_device


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_evaluation_has_no_training_side_effects() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "env=doorkey_6x6",
                "package=minimum",
                "smoke.enabled=true",
                "smoke.eval_episodes=1",
            ],
        )
    device = resolve_device("cpu")
    env = make_env(resolve_env_config(cfg), seed=0)
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(preprocess_obs(obs).shape[0], env.action_space.n, cfg.agent, device)
    before = {k: v.clone() for k, v in agent.online.state_dict().items()}

    rows = evaluate_agent(agent, cfg, [123], checkpoint_step=0, run_id="test")

    assert len(rows) == 1
    assert agent.optimizer_step_count == 0
    for key, value in agent.online.state_dict().items():
        assert torch.equal(value, before[key])
    env.close()
