#!/usr/bin/env python3
"""
Short-sighted (myopic) policy:
- 在每一步从若干候选动作中采样（multinomial over patterns）
- 对每个候选动作进行短期蒙特卡洛估计（在 env 的复制上做若干次 step）
- 选择期望即时成本最小的动作
注意：该策略计算量较大（取决于候选数与蒙特卡洛样本数），可调参数详见函数接口。
"""
import copy
import numpy as np

def select_myopic_action(env, theta=None, phi_fn=None, basis_params=None,
                         num_candidates=200, mc_samples=20, rng=None, x_max=None):
    """
    返回一个 action x (numpy array length env.n) 由 myopic 策略生成。

    参数:
      - env: CuttingStockEnv 实例（函数会使用 deepcopy(env) 来做模拟）
      - num_candidates: 候选动作数量（从 0..x_max 的 multinomial 采样）
      - mc_samples: 对每个候选评估时使用的蒙特卡洛样本数（越大越准但越慢）
      - rng: numpy RNG（若 None 则创建默认）
      - x_max: 最多切割的 patterns 总数（若 None 则使用 env.x_max）
    返回:
      - x_best: 一维 numpy int 数组长度 env.n
    """
    if rng is None:
        rng = np.random.default_rng()

    n = env.n
    if x_max is None:
        x_max = env.x_max

    best_x = np.zeros(n, dtype=int)
    best_score = float('inf')

    # sample candidate actions
    for _ in range(num_candidates):
        xtotal = int(rng.integers(0, x_max+1))
        if xtotal == 0:
            x = np.zeros(n, dtype=int)
        else:
            p = np.ones(n) / n
            x = rng.multinomial(xtotal, p)
        if not env.is_feasible(x):
            continue

        # monte-carlo estimate of immediate expected cost after applying x
        cost_accum = 0.0
        for _ in range(mc_samples):
            # deepcopy env to simulate demand draw and step without altering real env
            env_copy = copy.deepcopy(env)
            env_copy.reset(env.s)  # set same current inventory
            # apply action in the copy, env_copy.step will sample demand internally
            _, cost, _ = env_copy.step(x)
            cost_accum += cost
        avg_cost = cost_accum / float(mc_samples)
        if avg_cost < best_score:
            best_score = avg_cost
            best_x = x.copy()
    # if no feasible candidate found (rare), return zeros
    return best_x