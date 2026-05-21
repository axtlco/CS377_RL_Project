from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def env_key(cfg: Any) -> str:
    return str(cfg.env)


def resolve_env_config(cfg: Any):
    key = env_key(cfg)
    if key not in cfg.environments:
        choices = ", ".join(sorted(cfg.environments.keys()))
        raise KeyError(f"Unknown env={key!r}. Available environments: {choices}")
    data = OmegaConf.to_container(cfg.environments[key], resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Environment entry for {key!r} must be a mapping")
    data["key"] = key
    if cfg.env_max_steps is not None:
        data["max_steps"] = int(cfg.env_max_steps)
    return OmegaConf.create(data)
