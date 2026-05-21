from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SeedStream:
    seed: int
    stride: int = 100000

    def env_seed(self, episode_id: int) -> int:
        return self.seed * self.stride + episode_id

    def eval_seed(self, index: int) -> int:
        return (self.seed + 1) * self.stride + 50000 + index


def fixed_eval_seeds(seed: int, count: int) -> list[int]:
    stream = SeedStream(seed)
    return [stream.eval_seed(i) for i in range(count)]
