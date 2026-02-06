
# -*- coding: utf-8 -*-
"""
ilp_reuse_purelex.py

Extension of ilp_reuse.py:
- Adds "pure columns" (one per item type) as feasibility columns.
- Uses lexicographic (hierarchical) objectives:
    Phase-I  : minimize total pure bars  sum_i x_pure[i]
    Phase-II : minimize number of used real patterns sum_j y_real[j]
    Phase-III: minimize number of used pure patterns sum_i y_pure[i]  (tie-break)

Coverage is >= (overproduction allowed). Extra production is waste (not inventory).

Intended use: called inside rl_rollout_rc.py to avoid frequent infeasibility of the small pool ILP.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# --- helpers -------------------------------------------------

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
        Uj = int(np.min(d[pos] // col[pos]))
        U[j] = max(0, Uj)
    return U

def _build_pure_columns(w: np.ndarray, d: np.ndarray, L: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build m pure columns, each cuts as many items of type i as possible:
        a_i = min(d_i, floor(L / w_i)), at least 1 if d_i>0 and w_i<=L else 0.
    Returns:
        A_pure: [m, m]
        uids_pure: list[str] length m
        U_pure: upper bound on x_pure (<= ceil(d_i / a_i) if a_i>0 else 0)
    """
    w = np.asarray(w, dtype=int).reshape(-1)
    d = np.asarray(d, dtype=int).reshape(-1)
    m = d.size
    A_pure = np.zeros((m, m), dtype=int)
    uids = []
    U = np.zeros(m, dtype=int)
    for i in range(m):
        if int(d[i]) <= 0 or int(w[i]) <= 0 or int(w[i]) > int(L):
            ai = 0
        else:
            ai = int(L) // int(w[i])
            ai = min(int(d[i]), ai)
            ai = max(1, ai)  # if demand>0 and w<=L, keep at least 1
        A_pure[i, i] = ai
        uids.append(f"PURE[{i}]")
        if ai <= 0:
            U[i] = 0
        else:
            # minimal bars needed for item i is ceil(d_i/ai); this is a safe upper bound.
            U[i] = int((int(d[i]) + ai - 1) // ai)
    return A_pure, uids, U

# --- main solver --------------------------------------------

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

    # new: pure feasibility columns + lex objectives
    enable_pure: bool = True,
    w: Optional[np.ndarray] = None,
    L: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Solve a lexicographic ILP on a small column pool.

    Phase-I  : minimize total pure bars used  sum_i x_pure[i]
    Phase-II : minimize number of used real patterns sum_j y_real[j]
    Phase-III: minimize number of used pure patterns sum_i y_pure[i] (tie-break)

    Constraints:
        A_real x_real + A_pure x_pure >= d
        sum(x_real) + sum(x_pure) <= B0 + delta
        x_real[j] <= U_real[j] * y_real[j]
        x_pure[i] <= U_pure[i] * y_pure[i]
        x integer >=0; y binary
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
        U = np.maximum(U, (A.sum(axis=0) > 0).astype(int))
    else:
        U = np.asarray(U, dtype=int).reshape(-1)
        if U.size != n:
            raise ValueError("U length mismatch")

    # pure columns
    A_pure = None
    uids_pure: List[str] = []
    U_pure = None
    mpure = 0
    if enable_pure:
        if w is None or L is None:
            raise ValueError("enable_pure=True requires w and L")
        A_pure, uids_pure, U_pure = _build_pure_columns(w=w, d=d, L=int(L))
        mpure = int(A_pure.shape[1])

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise ImportError("gurobipy is required to solve the ε-constraint ILP") from e

    try:
        model = gp.Model("reuse_epsilon_ilp_purelex")
        if not verbose:
            model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", float(timelimit))
        if mipgap is not None:
            model.setParam("MIPGap", float(mipgap))
        if threads is not None:
            model.setParam("Threads", int(threads))

        # variables
        x_real = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")
        y_real = model.addVars(n, vtype=GRB.BINARY, name="y")

        if enable_pure:
            x_pure = model.addVars(mpure, vtype=GRB.INTEGER, lb=0, name="x_pure")
            y_pure = model.addVars(mpure, vtype=GRB.BINARY, name="y_pure")
        else:
            x_pure = None
            y_pure = None

        # coverage
        for i in range(m):
            expr = gp.quicksum(int(A[i, j]) * x_real[j] for j in range(n))
            if enable_pure:
                expr += gp.quicksum(int(A_pure[i, k]) * x_pure[k] for k in range(mpure))
            model.addConstr(expr >= int(d[i]), name=f"cov[{i}]")

        # bars cap
        bars_expr = gp.quicksum(x_real[j] for j in range(n))
        if enable_pure:
            bars_expr += gp.quicksum(x_pure[k] for k in range(mpure))
        model.addConstr(bars_expr <= bars_cap, name="bars_cap")

        # linking
        for j in range(n):
            model.addConstr(x_real[j] <= int(U[j]) * y_real[j], name=f"link[{j}]")
        if enable_pure:
            for k in range(mpure):
                model.addConstr(x_pure[k] <= int(U_pure[k]) * y_pure[k], name=f"link_pure[{k}]")

        # --- lexicographic objectives (hierarchical) ---
        # NOTE: Gurobi hierarchical objectives will optimize objectives by priority.
        # To avoid any degradation of higher-priority objectives, keep ObjNRelTol/ObjNAbsTol at 0.
        # (Defaults are typically 0, but we set explicitly for safety.)
        # See Gurobi docs for hierarchical/lexicographic multiobjective.
        # Phase-I: minimize pure bars
        if enable_pure:
            model.setObjectiveN(
                gp.quicksum(x_pure[k] for k in range(mpure)),
                index=0,
                priority=3,
                weight=1.0,
                name="phase1_min_pure_bars",
            )
            model.setAttr("ObjNRelTol", 0.0, 0)
            model.setAttr("ObjNAbsTol", 0.0, 0)
        else:
            # fall back: single objective patterns
            pass

        # Phase-II: minimize number of used real patterns
        model.setObjectiveN(
            gp.quicksum(y_real[j] for j in range(n)),
            index=1 if enable_pure else 0,
            priority=2,
            weight=1.0,
            name="phase2_min_real_patterns",
        )
        model.setAttr("ObjNRelTol", 0.0, 1 if enable_pure else 0)
        model.setAttr("ObjNAbsTol", 0.0, 1 if enable_pure else 0)

        # Phase-III: minimize number of used pure patterns (tie-break)
        if enable_pure:
            model.setObjectiveN(
                gp.quicksum(y_pure[k] for k in range(mpure)),
                index=2,
                priority=1,
                weight=1.0,
                name="phase3_min_pure_patterns",
            )
            model.setAttr("ObjNRelTol", 0.0, 2)
            model.setAttr("ObjNAbsTol", 0.0, 2)

        model.optimize()

        status = model.Status
        if status == GRB.OPTIMAL:
            status_s = "optimal"
        elif status == GRB.TIME_LIMIT:
            status_s = "timelimit"
        elif status == GRB.INFEASIBLE:
            return {
                "status": "infeasible",
                "bars_cap": int(bars_cap),
                "delta_used": int(delta),
                "enable_pure": bool(enable_pure),
            }
        else:
            return {
                "status": f"error_status_{int(status)}",
                "bars_cap": int(bars_cap),
                "delta_used": int(delta),
                "enable_pure": bool(enable_pure),
            }

        if model.SolCount <= 0:
            return {
                "status": status_s,
                "bars_cap": int(bars_cap),
                "delta_used": int(delta),
                "enable_pure": bool(enable_pure),
                "note": "no incumbent solution",
            }

        x_real_sol = np.array([int(round(x_real[j].X)) for j in range(n)], dtype=int)
        y_real_sol = (x_real_sol > 0).astype(int)

        if enable_pure:
            x_pure_sol = np.array([int(round(x_pure[k].X)) for k in range(mpure)], dtype=int)
            y_pure_sol = (x_pure_sol > 0).astype(int)
        else:
            x_pure_sol = np.zeros(0, dtype=int)
            y_pure_sol = np.zeros(0, dtype=int)

        bars_used = int(x_real_sol.sum() + x_pure_sol.sum())
        pure_bars_used = int(x_pure_sol.sum())
        real_bars_used = int(x_real_sol.sum())
        patterns_used_real = int(y_real_sol.sum())
        patterns_used_pure = int(y_pure_sol.sum())

        # best real pattern to propose for reuse (ignore pure)
        if patterns_used_real == 0:
            best_uid = None
            best_x = 0
        else:
            j_best = int(np.argmax(x_real_sol))
            best_uid = pattern_uids[j_best]
            best_x = int(x_real_sol[j_best])

        return {
            "status": status_s,
            "bars_cap": int(bars_cap),
            "bars_used": int(bars_used),
            "bars_used_real": int(real_bars_used),
            "bars_used_pure": int(pure_bars_used),
            "patterns_used_real": int(patterns_used_real),
            "patterns_used_pure": int(patterns_used_pure),
            "best_uid": best_uid,
            "best_x": int(best_x),

            # decision vectors
            "x_real": x_real_sol.tolist(),
            "y_real": y_real_sol.tolist(),
            "x_pure": x_pure_sol.tolist(),
            "y_pure": y_pure_sol.tolist(),
            "uids_real": list(pattern_uids),
            "uids_pure": list(uids_pure),

            "delta_used": int(delta),
            "enable_pure": bool(enable_pure),
        }

    except Exception as e:
        return {
            "status": "error",
            "bars_cap": int(bars_cap),
            "delta_used": int(delta),
            "enable_pure": bool(enable_pure),
            "error": str(e),
        }
