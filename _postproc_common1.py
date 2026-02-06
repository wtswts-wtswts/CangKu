# -*- coding: utf-8 -*-
"""
Common utilities for post-processing 1D-CSP solutions on a fixed pattern pool.

Assumptions (matches your pipeline outputs):
- pool.npy is an integer matrix P of shape (m, n): m item types, n patterns (columns).
- report.json contains:
    L, w (lengths), demand (list), (optional) other fields
- sequence.json contains steps with fields:
    pat_idx (>=0 index into pool columns) and k (bars used for that pattern)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import json
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = e
else:
    _GUROBI_IMPORT_ERROR = None


def load_pool(pool_path: str) -> np.ndarray:
    P = np.load(pool_path)
    if P.ndim != 2:
        raise ValueError(f"pool.npy must be 2D, got shape {P.shape}")
    # Ensure (m,n)
    if P.shape[0] < P.shape[1]:
        return P.astype(int)
    # Heuristic: if transposed, allow both
    return P.astype(int)


def load_report(report_path: str) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    if "demand" not in rep:
        raise ValueError("report.json must contain 'demand' list")
    if "L" not in rep:
        raise ValueError("report.json must contain 'L'")
    return rep


def load_sequence(sequence_path: str) -> List[Dict[str, Any]]:
    with open(sequence_path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_bars_from_sequence(sequence: List[Dict[str, Any]]) -> int:
    """
    Total number of bars used in a rollout sequence.
    IMPORTANT: counts steps even when pat_idx < 0 (e.g., final_mip with pat_idx=-1),
    because those bars are still real bars consumed.
    """
    total = 0
    for step in sequence:
        k = int(step.get("k", 0))
        if k > 0:
            total += k
    return int(total)


def incumbent_from_sequence(P: np.ndarray, sequence: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build incumbent x0,y0 on the pool columns from the RL sequence.
    If a step has pat_idx < 0, it is ignored (not in pool).
    """
    m, n = P.shape
    x0 = np.zeros(n, dtype=int)
    for step in sequence:
        j = int(step.get("pat_idx", -1))
        k = int(step.get("k", 0))
        if j >= 0 and j < n and k > 0:
            x0[j] += k
    y0 = (x0 > 0).astype(int)
    return x0, y0


def export_counts_from_xy(P: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
    """
    Export a counts.json-like structure:
      counts: { "a0,a1,...,am-1": bars_used }
      num_unique_patterns
      total_bars
    """
    m, n = P.shape
    x = np.asarray(x).reshape(-1)
    if x.shape[0] != n:
        raise ValueError("x length != n patterns")
    out_counts = {}
    for j in range(n):
        if int(x[j]) > 0:
            a = P[:, j].astype(int).tolist()
            key = ",".join(str(v) for v in a)
            out_counts[key] = out_counts.get(key, 0) + int(x[j])
    return {
        "counts": out_counts,
        "num_unique_patterns": int(np.sum(x > 0)),
        "total_bars": int(np.sum(x)),
    }


def build_base_model(P: np.ndarray, demand: np.ndarray, bars_ub: int):
    """
    Build min sum(y) s.t.
      P x >= demand
      sum(x) <= bars_ub
      x_j <= M y_j
    """
    if gp is None:
        raise RuntimeError(f"gurobipy not available: {_GUROBI_IMPORT_ERROR}")

    m, n = P.shape
    demand = np.asarray(demand, dtype=int).reshape(-1)
    if demand.shape[0] != m:
        raise ValueError("demand length != m")

    model = gp.Model("min_patterns_fixed_bars")
    model.Params.OutputFlag = 0

    x = model.addMVar(shape=n, vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addMVar(shape=n, vtype=GRB.BINARY, name="y")

    # cover constraints
    model.addConstr(P @ x >= demand, name="cover")

    # bar limit
    model.addConstr(x.sum() <= int(bars_ub), name="bars")

    # linking
    M = int(bars_ub)
    model.addConstr(x <= M * y, name="link")

    # objective: min patterns
    model.setObjective(y.sum(), GRB.MINIMIZE)

    return model, x, y


def set_start(model, x_vars, y_vars, x0: np.ndarray, y0: np.ndarray):
    """
    Set warm start.
    """
    x0 = np.asarray(x0, dtype=int).reshape(-1)
    y0 = np.asarray(y0, dtype=int).reshape(-1)
    for j in range(len(x0)):
        x_vars[j].Start = float(x0[j])
        y_vars[j].Start = float(y0[j])


def evaluate_xy(P: np.ndarray, rep: Dict[str, Any], x: np.ndarray) -> Dict[str, Any]:
    """
    Compute:
      bars = sum(x)
      demand_total_len = sum(w_i * demand_i)
      used_len (based on demand) = demand_total_len
      trim_loss_num = L*bars - demand_total_len
      trim_loss_pct = 100*trim_loss_num / demand_total_len
    """
    L = int(rep["L"])
    w = np.array(rep.get("w", []), dtype=int)
    demand = np.array(rep["demand"], dtype=int)
    if w.size == 0:
        # fallback: cannot compute length-based metrics without w
        return {"bars": int(np.sum(x)), "patterns": int(np.sum(np.asarray(x) > 0))}
    bars = int(np.sum(x))
    demand_total_len = int(np.dot(w, demand))
    trim_loss_num = int(L * bars - demand_total_len)
    trim_loss_pct = 100.0 * float(trim_loss_num) / float(demand_total_len) if demand_total_len > 0 else 0.0
    return {
        "bars": bars,
        "patterns": int(np.sum(np.asarray(x) > 0)),
        "demand_total_len": demand_total_len,
        "trim_loss_num": trim_loss_num,
        "trim_loss_pct": trim_loss_pct,
    }
