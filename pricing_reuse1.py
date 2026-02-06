# -*- coding: utf-8 -*-
"""
定价子问题（有界整数背包）模型复用模块（新增）
做了什么：
- 将 Gurobi 背包模型构建一次，并在每轮只更新目标系数（pi）再 optimize
- 这样可显著减少“重复建模”的开销，批量跑 Hard28/Scholl 时更快
- 注意：在严格 time limit 下，warm-start 可能导致搜索路径不同，但通常不改变最终 bars；
  如你追求完全一致，可固定 Gurobi Seed，并保留开关禁用复用。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import gurobipy as grb


@dataclass
class BoundedKnapsackPricer:
    w: np.ndarray
    L: int
    timelimit: float = 0.2
    seed: Optional[int] = None
    output_flag: int = 0

    def __post_init__(self):
        self.w = np.asarray(self.w, dtype=int)
        self.L = int(self.L)
        self.n = int(len(self.w))

        self.model = grb.Model("pricing_bounded_int_knapsack_reuse")
        self.model.Params.OutputFlag = int(self.output_flag)
        if self.timelimit is not None:
            self.model.Params.TimeLimit = float(self.timelimit)
        if self.seed is not None:
            # 固定随机性，便于复现
            self.model.Params.Seed = int(self.seed)

        self.a = self.model.addMVar(shape=self.n, vtype=grb.GRB.INTEGER, lb=0, name="a")

        # 约束：容量
        self.model.addConstr((self.w @ self.a) <= int(self.L), name="cap")

        # 需求上界约束：每次更新
        self.ub_constr = self.model.addConstr(self.a <= np.zeros(self.n, dtype=int), name="ub")  # placeholder

        # 目标：每次更新
        self.obj_expr = (np.zeros(self.n, dtype=float) @ self.a)
        self.model.setObjective(self.obj_expr, grb.GRB.MAXIMIZE)

        # 禁止过度清理模型，保证复用
        self.model.update()

    def solve(self, pi: np.ndarray, d_remain: Optional[np.ndarray], timelimit: Optional[float] = None) -> Tuple[np.ndarray, float]:
        pi = np.asarray(pi, dtype=float)
        if pi.size != self.n:
            raise ValueError(f"pi 维度不一致：{pi.size} != {self.n}")

        if d_remain is None:
            ub = np.full(self.n, fill_value=10**9, dtype=int)
        else:
            ub = np.asarray(d_remain, dtype=int).clip(min=0)

        if int(ub.sum()) == 0:
            return np.zeros(self.n, dtype=int), 0.0

        # 更新 time limit（可覆盖构造时的 timelimit）
        if timelimit is not None:
            self.model.Params.TimeLimit = float(timelimit)

        # 更新 ub 约束 RHS（Gurobi: 直接给 constr 的 RHS/或重建；这里用 setAttr 更稳）
        # 对 MVar 约束 a <= ub，这个约束是一个向量约束，Gurobi 会展开成多个 constr
        # 简单做法：删掉旧约束重建（仍然比重建整个模型轻）
        self.model.remove(self.ub_constr)
        self.ub_constr = self.model.addConstr(self.a <= ub, name="ub")
        self.model.update()

        # 更新目标：maximize pi @ a
        self.model.setObjective(pi @ self.a, grb.GRB.MAXIMIZE)

        self.model.optimize()
        if self.model.SolCount == 0:
            return np.zeros(self.n, dtype=int), 0.0

        sol = np.array(self.a.X, dtype=float).round().astype(int)
        sol = np.minimum(sol, ub)
        if int(np.inner(self.w, sol)) > int(self.L):
            return np.zeros(self.n, dtype=int), 0.0

        obj = float(np.inner(pi, sol))
        return sol.astype(int), obj