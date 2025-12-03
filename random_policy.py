#!/usr/bin/env python3
"""
Random policy: 从可行动作集合中随机采样
- 直接采样 xtotal 然后 multinomial 分配到 patterns
- 若采样得到的 x 不可行则重采直到找到或达到尝试上限
"""
import numpy as np

def select_random_action(env, rng=None, x_max=None, max_attempts=200):
    if rng is None:
        rng = np.random.default_rng()
    n = env.n
    if x_max is None:
        x_max = env.x_max

    for _ in range(max_attempts):
        xtotal = int(rng.integers(0, x_max+1))
        if xtotal == 0:
            x = np.zeros(n, dtype=int)
        else:
            p = np.ones(n) / n
            x = rng.multinomial(xtotal, p)
        if env.is_feasible(x):
            return x
    # fallback: zero action
    return np.zeros(n, dtype=int)