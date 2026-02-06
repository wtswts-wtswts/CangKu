# -*- coding: utf-8 -*-
"""
ilp_reuse.py

ε-constraint ILP for CSP-S style trade-off:
- Hard constraint on bars: sum_j x_j <= B0 + delta
- Objective: minimize number of used patterns: sum_j y_j
- Coverage: A x >= d

Designed to be called on a *small* pool of patterns (top-k from RL + selected history).
Solver: gurobipy (preferred). If gurobipy is unavailable, this module raises an ImportError
with a clear message.

Returned stats are JSON-serializable.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def _compute_U_bounds(A: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Tight per-pattern upper bound U_j = min_{i: a_ij>0} floor(d_i / a_ij), at least 0.
    If a column covers nothing, U_j=0.
    """
    m, n = A.shape
    d = np.asarray(d, dtype=int).reshape(-1)
    U = np.zeros(n, dtype=int)
    for j in range(n):
        col = A[:, j].astype(int)
        pos = col > 0
        if not np.any(pos):
            U[j] = 0
            continue
        U[j] = int(np.min(d[pos] // col[pos])) if int(np.min(d[pos] // col[pos])) >= 0 else 0
    return U


def solve_epsilon_ilp(
    A: np.ndarray,
    d: np.ndarray,
    B0: int,
    delta: int = 0,
    timelimit: float = 0.5,
    pattern_uids: Optional[List[str]] = None,
    U: Optional[np.ndarray] = None,
    mipgap: Optional[float] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve:
        min sum y_j
        s.t. A x >= d
             sum x <= B0 + delta
             x_j <= U_j y_j
             x integer >=0, y binary

    Inputs:
        A: shape [m, n]
        d: shape [m]
        B0: baseline bars (typically ceil LP objective)
        delta: slack on bars constraint
        pattern_uids: optional list of ids of length n
        U: optional upper bounds, length n; if None, computed from A and d

    Returns dict:
        status: 'optimal'|'timelimit'|'infeasible'|'error'
        bars_cap, bars_used, patterns_used
        best_uid, best_x
        x (optional, list[int]) only if feasible
        y (optional, list[int]) only if feasible
        delta_used
    """
    A = np.asarray(A, dtype=int)
    d = np.asarray(d, dtype=int).reshape(-1)
    assert A.shape[0] == d.size, "A and d dimension mismatch"
    m, n = A.shape
    if pattern_uids is None:
        pattern_uids = [str(j) for j in range(n)]
    if len(pattern_uids) != n:
        raise ValueError("pattern_uids length mismatch")

    bars_cap = int(B0) + int(delta)
    if bars_cap < 0:
        raise ValueError("bars_cap < 0")

    if U is None:
        U = _compute_U_bounds(A, d)
        # In practice, allow at least 1 if a column covers something; otherwise y could never activate.
        U = np.maximum(U, (A.sum(axis=0) > 0).astype(int))
    else:
        U = np.asarray(U, dtype=int).reshape(-1)
        if U.size != n:
            raise ValueError("U length mismatch")

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("gurobipy is required to solve the ε-constraint ILP") from e

    try:
        model = gp.Model("reuse_epsilon_ilp")
        if not verbose:
            model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", float(timelimit))
        if mipgap is not None:
            model.setParam("MIPGap", float(mipgap))
        if threads is not None:
            model.setParam("Threads", int(threads))

        x = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")
        y = model.addVars(n, vtype=GRB.BINARY, name="y")

        # coverage
        for i in range(m):
            expr = gp.quicksum(int(A[i, j]) * x[j] for j in range(n))
            model.addConstr(expr >= int(d[i]), name=f"cov[{i}]")

        # bars cap
        model.addConstr(gp.quicksum(x[j] for j in range(n)) <= bars_cap, name="bars_cap")

        # linking
        for j in range(n):
            model.addConstr(x[j] <= int(U[j]) * y[j], name=f"link[{j}]")

        # objective: minimize patterns
        model.setObjective(gp.quicksum(y[j] for j in range(n)), GRB.MINIMIZE)

        model.optimize()

        status = model.Status
        if status == GRB.OPTIMAL:
            status_s = "optimal"
        elif status == GRB.TIME_LIMIT:
            # could still have a feasible solution
            status_s = "timelimit"
        elif status == GRB.INFEASIBLE:
            return {
                "status": "infeasible",
                "bars_cap": bars_cap,
                "delta_used": int(delta),
            }
        else:
            return {
                "status": f"error_status_{int(status)}",
                "bars_cap": bars_cap,
                "delta_used": int(delta),
            }

        if model.SolCount <= 0:
            return {
                "status": status_s,
                "bars_cap": bars_cap,
                "delta_used": int(delta),
                "note": "no incumbent solution",
            }

        x_sol = np.array([int(round(x[j].X)) for j in range(n)], dtype=int)
        y_sol = (x_sol > 0).astype(int)
        bars_used = int(x_sol.sum())
        patterns_used = int(y_sol.sum())

        if patterns_used == 0:
            best_uid = None
            best_x = 0
        else:
            j_best = int(np.argmax(x_sol))
            best_uid = pattern_uids[j_best]
            best_x = int(x_sol[j_best])

        return {
            "status": status_s,
            "bars_cap": int(bars_cap),
            "bars_used": int(bars_used),
            "patterns_used": int(patterns_used),
            "best_uid": best_uid,
            "best_x": int(best_x),
            "x": x_sol.tolist(),
            "y": y_sol.tolist(),
            "delta_used": int(delta),
        }

    except Exception as e:
        return {
            "status": "error",
            "bars_cap": int(bars_cap),
            "delta_used": int(delta),
            "error": str(e),
        }
