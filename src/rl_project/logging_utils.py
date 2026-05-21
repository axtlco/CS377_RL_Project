from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .run_metadata import collect_run_metadata


class RunLogger:
    def __init__(self, run_dir: str | Path, cfg: Any) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir = self.run_dir / "tables"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.tables_dir.mkdir(exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, self.run_dir / "resolved_config.yaml")
        metadata = collect_run_metadata(cfg)
        (self.run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.scalar_path = self.run_dir / "scalars.csv"
        self._scalar_file = self.scalar_path.open("a", newline="")
        self._scalar_writer = csv.DictWriter(self._scalar_file, fieldnames=["step", "name", "value"])
        if self.scalar_path.stat().st_size == 0:
            self._scalar_writer.writeheader()
        self.tb = SummaryWriter(str(self.run_dir / "tensorboard")) if bool(cfg.logging.tensorboard) else None
        self.episode_rows: list[dict[str, Any]] = []
        self.eval_rows: list[dict[str, Any]] = []

    def scalar(self, name: str, value: float, step: int) -> None:
        self._scalar_writer.writerow({"step": step, "name": name, "value": value})
        self._scalar_file.flush()
        if self.tb is not None:
            self.tb.add_scalar(name, value, step)

    def episode(self, row: dict[str, Any]) -> None:
        self.episode_rows.append(row)

    def eval(self, rows: list[dict[str, Any]]) -> None:
        self.eval_rows.extend(rows)

    def flush_tables(self) -> None:
        if self.episode_rows:
            pd.DataFrame(self.episode_rows).to_parquet(self.tables_dir / "episodes.parquet", index=False)
            pd.DataFrame(self.episode_rows).to_csv(self.tables_dir / "episodes.csv", index=False)
        if self.eval_rows:
            pd.DataFrame(self.eval_rows).to_parquet(self.tables_dir / "evaluation.parquet", index=False)
            pd.DataFrame(self.eval_rows).to_csv(self.tables_dir / "evaluation.csv", index=False)

    def close(self) -> None:
        self.flush_tables()
        if self.tb is not None:
            self.tb.close()
        self._scalar_file.close()
