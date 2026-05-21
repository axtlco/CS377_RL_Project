from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from rl_project.offline_train import run_offline_training
from rl_project.trainer import run_training


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.mark.parametrize("algorithm", ["dqn_1step", "dqn_nstep"])
def test_short_online_smoke_creates_checkpoint_logs_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, algorithm: str
) -> None:
    work_dir = tmp_path / algorithm
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={algorithm}",
                "package=minimum",
                "env=doorkey_6x6",
                "seed=0",
                "device=cpu",
                "env_max_steps=20",
                "smoke.enabled=true",
                "smoke.training_steps=40",
                "smoke.eval_interval=20",
                "smoke.eval_episodes=1",
                "smoke.learning_starts=8",
                "smoke.batch_size=8",
                "logging.tensorboard=false",
            ],
        )

    run_dir = run_training(cfg)

    assert (run_dir / "checkpoints" / "step_40.pt").exists()
    assert (run_dir / "scalars.csv").exists()
    assert (run_dir / "tables" / "evaluation.parquet").exists()


def test_fixed_replay_offline_smoke_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "offline"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=dqn_nstep",
                "package=minimum",
                "replay=fixed_replay",
                "env=doorkey_6x6",
                "seed=0",
                "device=cpu",
                "replay.episode_count=1",
                "replay.collector_policy=mixed",
                "replay.target_success_episodes=1",
                "replay.offline_updates=4",
                "replay.diagnostic_interval=1",
                "smoke.enabled=true",
                "smoke.batch_size=4",
                "logging.tensorboard=false",
            ],
        )

    run_dir = run_offline_training(cfg)

    assert (run_dir / "fixed_replay" / "metadata.json").exists()
    assert (run_dir / "checkpoints" / "offline_final.pt").exists()
    assert (run_dir / "tables" / "offline_success_prefix_diagnostics.parquet").exists()
