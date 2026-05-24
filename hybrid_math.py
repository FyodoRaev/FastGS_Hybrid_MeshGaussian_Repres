from __future__ import annotations

import math


def annealed_beta(iteration: int, beta_min: float, beta_max: float, warmup_iters: int) -> float:
    if warmup_iters <= 0 or beta_min <= 0 or beta_max <= 0:
        return float(beta_max)

    t = min(max(iteration, 0), warmup_iters) / float(warmup_iters)
    return float(beta_min * math.pow(beta_max / beta_min, t))
