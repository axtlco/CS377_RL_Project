from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ALGORITHM_FACTORS = {
    "dqn_1step": {"ride": False, "backup": "1step"},
    "dqn_nstep": {"ride": False, "backup": "nstep"},
    "dqn_ride_1step": {"ride": True, "backup": "1step"},
    "dqn_ride_nstep": {"ride": True, "backup": "nstep"},
}


RUN_KEYS = ["run_dir", "run_id", "algorithm", "package", "env_id", "seed"]
CURVE_KEYS = [*RUN_KEYS, "checkpoint_step"]
AGG_KEYS = ["package", "env_id", "algorithm"]


def discover_run_dirs(paths: Iterable[str | Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if (path / "tables").exists():
            run_dirs.add(path)
            continue
        for table_path in path.rglob("tables/evaluation.*"):
            run_dirs.add(table_path.parents[1])
    return sorted(run_dirs)


def load_table(run_dir: Path, name: str) -> pd.DataFrame:
    parquet_path = run_dir / "tables" / f"{name}.parquet"
    csv_path = run_dir / "tables" / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_runs(run_dirs: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_frames = []
    episode_frames = []
    for run_dir in run_dirs:
        evaluation = load_table(run_dir, "evaluation")
        episodes = load_table(run_dir, "episodes")
        if not evaluation.empty:
            evaluation = evaluation.copy()
            evaluation["run_dir"] = str(run_dir)
            eval_frames.append(evaluation)
        if not episodes.empty:
            episodes = episodes.copy()
            episodes["run_dir"] = str(run_dir)
            episode_frames.append(episodes)
    eval_rows = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    episode_rows = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    return eval_rows, episode_rows


def checkpoint_eval_curve(eval_rows: pd.DataFrame) -> pd.DataFrame:
    if eval_rows.empty:
        return pd.DataFrame()
    return (
        eval_rows.groupby(CURVE_KEYS, dropna=False)
        .agg(
            success_rate=("success", "mean"),
            return_ext_mean=("return_ext", "mean"),
            episode_length_mean=("episode_length", "mean"),
            eval_episodes=("success", "size"),
        )
        .reset_index()
        .sort_values(CURVE_KEYS)
    )


def summarize_runs(eval_curve: pd.DataFrame, episode_rows: pd.DataFrame) -> pd.DataFrame:
    if eval_curve.empty:
        return pd.DataFrame()
    episode_groups = {key: frame for key, frame in episode_rows.groupby("run_dir")} if not episode_rows.empty else {}
    rows = []
    for keys, curve in eval_curve.groupby(RUN_KEYS, dropna=False):
        curve = curve.sort_values("checkpoint_step")
        final = curve.iloc[-1]
        episode_frame = episode_groups.get(keys[0], pd.DataFrame())
        first_success = first_event_global_step(episode_frame, "first_success_step")
        row = {
            **dict(zip(RUN_KEYS, keys)),
            "final_checkpoint_step": int(final["checkpoint_step"]),
            "final_success_rate": float(final["success_rate"]),
            "final_return_ext_mean": float(final["return_ext_mean"]),
            "best_success_rate": float(curve["success_rate"].max()),
            "success_rate_auc": normalized_auc(curve["checkpoint_step"], curve["success_rate"]),
            "first_key_global_step": first_event_global_step(episode_frame, "first_key_step"),
            "first_door_global_step": first_event_global_step(episode_frame, "first_door_step"),
            "first_room_global_step": first_event_global_step(episode_frame, "first_room_step"),
            "first_success_global_step": first_success,
            "train_success_episodes": int(episode_frame["success"].sum()) if "success" in episode_frame else 0,
            "episode_count": int(len(episode_frame)),
            "post_discovery_to_25_success": post_discovery_threshold_step(curve, first_success, 0.25),
            "post_discovery_to_50_success": post_discovery_threshold_step(curve, first_success, 0.50),
            "post_discovery_to_75_success": post_discovery_threshold_step(curve, first_success, 0.75),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def normalized_auc(steps: pd.Series, values: pd.Series) -> float:
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x[0] > 0:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([0.0], y))
    total = max(float(x[-1]), 1.0)
    return float(np.trapezoid(y, x) / total)


def first_event_global_step(episodes: pd.DataFrame, event_step_column: str) -> float:
    if episodes.empty or event_step_column not in episodes:
        return float("nan")
    event_rows = episodes[episodes[event_step_column].notna()].copy()
    if event_rows.empty:
        return float("nan")
    event_steps = (
        event_rows["global_step"].astype(float)
        - event_rows["episode_length"].astype(float)
        + event_rows[event_step_column].astype(float)
    )
    return float(event_steps.min())


def post_discovery_threshold_step(curve: pd.DataFrame, first_success_step: float, threshold: float) -> float:
    if np.isnan(first_success_step):
        return float("nan")
    eligible = curve[(curve["checkpoint_step"] >= first_success_step) & (curve["success_rate"] >= threshold)]
    if eligible.empty:
        return float("nan")
    return float(eligible["checkpoint_step"].min() - first_success_step)


def aggregate_summary(run_summary: pd.DataFrame, bootstrap_samples: int = 1000) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    metric_columns = [
        "final_success_rate",
        "best_success_rate",
        "success_rate_auc",
        "first_key_global_step",
        "first_door_global_step",
        "first_success_global_step",
        "post_discovery_to_50_success",
    ]
    rows = []
    for keys, frame in run_summary.groupby(AGG_KEYS, dropna=False):
        for metric in metric_columns:
            values = frame[metric].dropna().to_numpy(dtype=np.float64)
            low, high = bootstrap_mean_ci(values, bootstrap_samples)
            rows.append(
                {
                    **dict(zip(AGG_KEYS, keys)),
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "median": float(np.median(values)) if values.size else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, samples: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or samples <= 0:
        value = float(np.mean(values))
        return value, value
    rng = np.random.default_rng(0)
    means = []
    for _ in range(samples):
        draw = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(draw)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def factorial_effects(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    frame = run_summary[run_summary["algorithm"].isin(ALGORITHM_FACTORS)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["ride"] = frame["algorithm"].map(lambda name: ALGORITHM_FACTORS[name]["ride"])
    frame["backup"] = frame["algorithm"].map(lambda name: ALGORITHM_FACTORS[name]["backup"])
    metrics = ["final_success_rate", "best_success_rate", "success_rate_auc"]
    rows = []
    for keys, group in frame.groupby(["package", "env_id"], dropna=False):
        means = group.groupby(["backup", "ride"])[metrics].mean()
        required = [("1step", False), ("1step", True), ("nstep", False), ("nstep", True)]
        if not all(index in means.index for index in required):
            continue
        for metric in metrics:
            y00 = float(means.loc[("1step", False), metric])
            y10 = float(means.loc[("1step", True), metric])
            y01 = float(means.loc[("nstep", False), metric])
            y11 = float(means.loc[("nstep", True), metric])
            rows.append(
                {
                    "package": keys[0],
                    "env_id": keys[1],
                    "metric": metric,
                    "dqn_1step": y00,
                    "dqn_ride_1step": y10,
                    "dqn_nstep": y01,
                    "dqn_ride_nstep": y11,
                    "ride_main_effect": ((y10 + y11) / 2.0) - ((y00 + y01) / 2.0),
                    "nstep_main_effect": ((y01 + y11) / 2.0) - ((y00 + y10) / 2.0),
                    "ride_x_nstep_interaction": (y11 - y01) - (y10 - y00),
                }
            )
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{name}.csv", index=False)
    try:
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    except ImportError:
        pass


def run_analysis(paths: list[str | Path], output_dir: str | Path, bootstrap_samples: int = 1000) -> Path:
    output_dir = Path(output_dir)
    run_dirs = discover_run_dirs(paths)
    eval_rows, episode_rows = load_runs(run_dirs)
    eval_curve = checkpoint_eval_curve(eval_rows)
    run_summary = summarize_runs(eval_curve, episode_rows)
    aggregate = aggregate_summary(run_summary, bootstrap_samples)
    effects = factorial_effects(run_summary)
    write_table(eval_curve, output_dir, "checkpoint_eval")
    write_table(run_summary, output_dir, "run_summary")
    write_table(aggregate, output_dir, "aggregate_summary")
    write_table(effects, output_dir, "factorial_effects")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RL project experiment outputs.")
    parser.add_argument("paths", nargs="+", help="Run directories or parent output directories to scan.")
    parser.add_argument("--out", default="analysis", help="Directory for analysis CSV/Parquet tables.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    output_dir = run_analysis(args.paths, args.out, args.bootstrap_samples)
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
