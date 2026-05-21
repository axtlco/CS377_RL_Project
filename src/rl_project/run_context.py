from __future__ import annotations

from pathlib import Path

from hydra.core.hydra_config import HydraConfig


def resolve_output_dir() -> Path:
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd() / "outputs" / "manual"
