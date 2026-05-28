from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_minimum_package_overrides_expected_values() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["package=minimum"])

    assert cfg.package.envs == ["doorkey_6x6", "doorkey_8x8", "doorkey_16x16"]
    assert cfg.package.paired_seeds == 10
    assert cfg.package.training_steps.doorkey_8x8 == 1000000
    assert cfg.package.eval_episodes.doorkey_6x6 == 100


def test_full_package_overrides_expected_values() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["package=full"])

    assert "doorkey_12x12" in cfg.package.envs
    assert cfg.package.paired_seeds == 30
    assert cfg.package.training_steps.doorkey_12x12 == 5000000
    assert cfg.package.eval_episodes.doorkey_16x16 == 500


def test_ride_algorithm_config_enables_ride() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["algorithm=dqn_ride_nstep"])

    assert cfg.algorithm.name == "dqn_ride_nstep"
    assert cfg.algorithm.n_step == 3
    assert cfg.ride.enabled
