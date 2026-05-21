from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from rl_project.datasets import collect_fixed_replay, load_replay_rows


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_scripted_doorkey_collector_creates_successful_episode(tmp_path: Path) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "package=minimum",
                "replay=fixed_replay",
                "env=doorkey_6x6",
                "seed=0",
                "replay.collector_policy=scripted_success",
                "replay.episode_count=1",
            ],
        )

    data_path = collect_fixed_replay(cfg, tmp_path)
    rows = load_replay_rows(data_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert any(row["reached_goal"] for row in rows)
    assert metadata["successful_episode_count"] == 1
    assert metadata["success_ratio"] == 1.0


def test_mixed_fixed_replay_controls_success_count_and_metadata(tmp_path: Path) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "package=minimum",
                "replay=fixed_replay",
                "env=doorkey_6x6",
                "seed=0",
                "replay.collector_policy=mixed",
                "replay.episode_count=2",
                "replay.target_success_episodes=1",
            ],
        )

    collect_fixed_replay(cfg, tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["dataset_id"]
    assert metadata["env_id"] == "doorkey_6x6"
    assert metadata["dataset_seed"] == 0
    assert metadata["target_success_ratio"] is None
    assert metadata["episode_count"] == 2
    assert metadata["transition_count"] > 0
    assert metadata["successful_episode_count"] == 1
    assert metadata["success_ratio"] == 0.5
    assert metadata["collector_policy"] == "mixed"
    assert metadata["collector_episode_counts"]["scripted_success"] == 1
    assert metadata["schema_version"] == 1
