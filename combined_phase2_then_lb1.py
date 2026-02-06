# -*- coding: utf-8 -*-
"""
combined_phase2_then_lb.py

Stage-2 post-processing for 1D Cutting Stock (single stock length, unlimited bars).

Inputs (from stage1):
  - pool.npy      : pattern matrix A (m x P)
  - report.json   : instance meta (L, w, demand/d, stage1 bars/patterns/trimloss, optional x)
  - sequence.json : stage1 chosen actions/pattern indices (optional, used to reconstruct x_start)

Modes:
  - none     : print stage1 summary
  - phase2   : solve Phase-2 MIP (reduce #patterns) with bars upper bound
  - lb       : run enhanced Local Branching to further reduce #patterns
  - combined : phase2 then lb

NO stage3 in this script.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


# -------------------------
# helpers
# -------------------------

def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise RuntimeError("gurobipy not available")

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def demand_based_trimloss_pct(w: np.ndarray, d: np.ndarray, L: int, bars: int) -> float:
    """
    Demand-based trim loss (same style you used before):
      100 * (bars*L - sum_i w_i*d_i) / sum_i w_i*d_i
    """
    w = np.asarray(w, dtype=int).reshape(-1)
    d = np.asarray(d, dtype=int).reshape(-1)
    demand_sum = float(np.sum(w * d))
    if demand_sum <= 0:
        return 0.0
    return 100.0 * (float(L) * float(bars) - demand_sum) / demand_sum

def realized_trimloss_pct(w: np.ndarray, x: np.ndarray, A: np.ndarray, L: int) -> float:
    """
    Realized trim loss based on chosen patterns x:
      total_trim / total_stock * 100
      total_trim  = sum_j x_j * (L - sum_i w_i*A[i,j])
      total_stock = (sum_j x_j) * L
    """
    w = np.asarray(w, dtype=int).reshape(-1)
    x = np.asarray(x, dtype=int).reshape(-1)
    A = np.asarray(A, dtype=int)
    bars = int(np.sum(x))
    if bars <= 0:
        return 0.0

    trims = []
    for j in range(A.shape[1]):
        if x[j] <= 0:
            trims.append(0.0)
            continue
        used = int(np.dot(w, A[:, j]))
        trim = max(0, int(L) - used)
        trims.append(float(trim))

    total_trim = float(np.dot(x.astype(float), np.array(trims, dtype=float)))
    total_stock = float(bars) * float(L)
    if total_stock <= 0:
        return 0.0
    return 100.0 * total_trim / total_stock

def pattern_key(a_col: np.ndarray) -> str:
    return ",".join(str(int(v)) for v in a_col.astype(int).tolist())

def counts_from_x(A: np.ndarray, x: np.ndarray) -> Dict[str, int]:
    A = np.asarray(A, dtype=int)
    x = np.asarray(x, dtype=int).reshape(-1)
    out: Dict[str, int] = {}
    for j in range(A.shape[1]):
        if x[j] <= 0:
            continue
        k = int(x[j])
        key = pattern_key(A[:, j])
        out[key] = out.get(key, 0) + k
    return out

def summarize_solution(A: np.ndarray, x: np.ndarray, w: np.ndarray, d: np.ndarray, L: int) -> Dict[str, Any]:
    x = np.asarray(x, dtype=int).reshape(-1)
    total_bars = int(np.sum(x))
    num_unique = int(np.sum(x > 0))
    return {
        "total_bars": total_bars,
        "num_unique_patterns": num_unique,
        "trim_loss_pct_demand": float(demand_based_trimloss_pct(w, d, L, max(1, total_bars))),
        "trim_loss_pct_realized": float(realized_trimloss_pct(w, x, A, L)),
        "counts": counts_from_x(A, x),
    }


# -------------------------
# Phase-2 MIP: reduce patterns with bars upper bound
# -------------------------

def solve_phase2_pattern_reduction(
    A: np.ndarray,
    d: np.ndarray,
    bars_ub: int,
    timelimit: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    min   sum_j y_j
    s.t.  A x >= d
          sum_j x_j <= bars_ub
          0 <= x_j integer
          x_j <= M y_j, y_j binary
    """
    _require_gurobi()

    A = np.asarray(A, dtype=int)
    d = np.asarray(d, dtype=int).reshape(-1)
    m, P = A.shape
    if d.shape[0] != m:
        raise ValueError("demand length != m")

    model = gp.Model("phase2_reduce_patterns")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = float(timelimit)

    x = model.addMVar(shape=P, vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addMVar(shape=P, vtype=GRB.BINARY, name="y")

    model.addConstr(A @ x >= d, name="cover")
    model.addConstr(x.sum() <= int(bars_ub), name="bars_ub")

    M = int(max(1, bars_ub))
    model.addConstr(x <= M * y, name="link")

    model.setObjective(y.sum(), GRB.MINIMIZE)
    model.optimize()

    status = int(model.Status)
    out: Dict[str, Any] = {"status": status, "obj_patterns": None, "x": None}

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        try:
            out["obj_patterns"] = float(model.ObjVal)
            out["x"] = np.rint(x.X).astype(int).tolist()
        except Exception:
            pass
    return out


# -------------------------
# Enhanced Local Branching on y (binaries)
#   Δ(y, y*) <= k (intensification)
#   if improved: add outside cut Δ(y, y_old) >= k+1 (diversification)
#   else: increase k (adaptive)
#   warm start every iteration with incumbent
# -------------------------

def hamming_expr(y_vars: List[Any], y_ref: np.ndarray) -> "gp.LinExpr":
    """
    Δ(y, y_ref) = sum_{j:y_ref=1}(1-y_j) + sum_{j:y_ref=0} y_j
    """
    expr = gp.LinExpr()
    for j, v in enumerate(y_vars):
        if int(y_ref[j]) == 1:
            expr += (1 - v)
        else:
            expr += v
    return expr

@dataclass
class LBEnhancedResult:
    best_obj_patterns: int
    best_x: np.ndarray
    traces: List[Dict[str, Any]]
    outside_cuts: int

def solve_lb_enhanced(
    A: np.ndarray,
    d: np.ndarray,
    bars_ub: int,
    x_start: np.ndarray,
    k_list: List[int],
    time_per_k: float,
    verbose: bool = False,
) -> LBEnhancedResult:
    _require_gurobi()

    A = np.asarray(A, dtype=int)
    d = np.asarray(d, dtype=int).reshape(-1)
    x_start = np.asarray(x_start, dtype=int).reshape(-1)

    m, P = A.shape
    if d.shape[0] != m:
        raise ValueError("demand length != m")
    if x_start.shape[0] != P:
        raise ValueError("x_start length != #patterns")

    k_list = [int(k) for k in k_list if str(k).strip() != ""]
    if not k_list:
        k_list = [3, 5, 8, 12]
    k0 = int(k_list[0])
    k_max = int(max(k_list))

    # incumbent
    best_x = x_start.copy()
    best_y = (best_x > 0).astype(int)
    best_obj = int(best_y.sum())

    # one reusable base model
    model = gp.Model("lb_enhanced_reduce_patterns")
    model.Params.OutputFlag = 1 if verbose else 0

    x = model.addVars(P, vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addVars(P, vtype=GRB.BINARY, name="y")

    M = int(max(1, bars_ub))
    for j in range(P):
        model.addConstr(x[j] <= M * y[j], name=f"link_{j}")

    for i in range(m):
        model.addConstr(gp.quicksum(int(A[i, j]) * x[j] for j in range(P)) >= int(d[i]), name=f"demand_{i}")

    model.addConstr(gp.quicksum(x[j] for j in range(P)) <= int(bars_ub), name="bars_ub")
    model.setObjective(gp.quicksum(y[j] for j in range(P)), GRB.MINIMIZE)

    # constraints bookkeeping
    in_constr = None
    outside_refs: List[Tuple[np.ndarray, int]] = []  # (y_old, k_used)

    traces: List[Dict[str, Any]] = []
    k = k0
    no_improve = 0
    t_global = time.time()

    # total time budget = len(k_list)*time_per_k (keeps your old CLI semantics)
    time_budget = float(max(0.1, len(k_list) * float(time_per_k)))

    it = 0
    while True:
        it += 1
        elapsed = time.time() - t_global
        if elapsed >= time_budget:
            break

        # remove previous in-neighborhood cut
        if in_constr is not None:
            try:
                model.remove(in_constr)
                model.update()
            except Exception:
                pass
            in_constr = None

        # add current in-neighborhood cut Δ(y, best_y) <= k
        expr_in = hamming_expr([y[j] for j in range(P)], best_y)
        in_constr = model.addConstr(expr_in <= int(k), name=f"lb_in_{it}")

        # add outside cuts once
        for idx, (y_old, k_used) in enumerate(outside_refs):
            cname = f"lb_out_{idx}"
            if model.getConstrByName(cname) is None:
                expr_out = hamming_expr([y[j] for j in range(P)], y_old)
                model.addConstr(expr_out >= int(k_used) + 1, name=cname)

        model.update()

        # warm start from incumbent
        for j in range(P):
            x[j].Start = int(best_x[j])
            y[j].Start = int(best_y[j])

        # solve
        model.Params.TimeLimit = float(max(0.05, time_per_k))
        model.optimize()

        status = int(model.Status)
        rec: Dict[str, Any] = {
            "it": int(it),
            "k": int(k),
            "elapsed": float(time.time() - t_global),
            "status": status,
            "obj": None,
            "best_obj": int(best_obj),
            "improved": False,
            "outside_cuts": int(len(outside_refs)),
        }

        improved = False
        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            x_val = np.array([int(round(x[j].X)) for j in range(P)], dtype=int)
            y_val = (x_val > 0).astype(int)
            obj = int(y_val.sum())
            rec["obj"] = int(obj)
            if obj < best_obj:
                # improvement -> external branching: forbid returning to old neighborhood
                old_y = best_y.copy()
                old_k = int(k)

                best_obj = int(obj)
                best_x = x_val
                best_y = y_val

                outside_refs.append((old_y, old_k))
                improved = True
                rec["improved"] = True
                rec["best_obj"] = int(best_obj)

                # reset neighborhood
                k = k0
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        traces.append(rec)

        if improved:
            continue

        # adapt k upward if no improvement
        higher = [kk for kk in k_list if kk > k]
        if higher:
            k = int(min(higher))
        else:
            k = int(k + max(1, k // 2))

        if k > k_max:
            k = k_max
            # if already stuck at k_max for 2 iterations -> stop
            if no_improve >= 2:
                break

    return LBEnhancedResult(
        best_obj_patterns=int(best_obj),
        best_x=best_x,
        traces=traces,
        outside_cuts=int(len(outside_refs)),
    )


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "phase2", "lb", "combined"], default="combined")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sequence", required=True)

    parser.add_argument("--bars-delta", type=int, default=0)
    parser.add_argument("--phase2-timelimit", type=float, default=0.5)

    parser.add_argument("--k-list", type=str, default="3,5,8,12")
    parser.add_argument("--time-per-k", type=float, default=1.0)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    A = np.load(args.pool)
    report = load_json(args.report)
    seq = load_json(args.sequence)

    # instance meta
    w = np.asarray(report["w"], dtype=int)
    d_raw = report.get("demand", report.get("d", None))
    if d_raw is None:
        raise KeyError('report.json missing demand vector: expected key "demand" (stage1) or "d"')
    d = np.asarray(d_raw, dtype=int)
    L = int(report["L"])

    # bars upper bound
    stage1_bars = int(report.get("bars", report.get("bars_ub", 0)))
    bars_ub = int(stage1_bars + int(args.bars_delta))

    # stage1 metrics (if present)
    stage1_patterns = int(report.get("patterns", report.get("num_patterns", -1)))
    stage1_trim = float(report.get("trim_loss_pct", demand_based_trimloss_pct(w, d, L, max(1, stage1_bars))))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "none":
        print(f"[Stage1] bars={stage1_bars}, patterns={stage1_patterns}, trimloss(demand-based)={stage1_trim:.6f}%")
        return

    # reconstruct x_start
    P = A.shape[1]
    x_start = np.zeros(P, dtype=int)

    if "x" in report and isinstance(report["x"], list) and len(report["x"]) == P:
        x_start = np.asarray(report["x"], dtype=int)
    else:
        for step in seq:
            pat_idx = step.get("pat_idx", None)
            k = int(step.get("k", 1))
            if pat_idx is None:
                continue
            j = int(pat_idx)
            if 0 <= j < P:
                x_start[j] += int(k)

    # Phase2
    x_phase2 = x_start.copy()
    phase2_res = None
    if args.mode in ("phase2", "combined"):
        phase2_res = solve_phase2_pattern_reduction(
            A=A, d=d, bars_ub=bars_ub, timelimit=float(args.phase2_timelimit), verbose=args.verbose
        )
        if phase2_res.get("x") is not None:
            x_phase2 = np.asarray(phase2_res["x"], dtype=int)

    # LB (enhanced)
    x_best = x_phase2.copy()
    lb_res: Optional[Dict[str, Any]] = None
    if args.mode in ("lb", "combined"):
        k_list = [int(s.strip()) for s in str(args.k_list).split(",") if s.strip()]
        lb = solve_lb_enhanced(
            A=A, d=d, bars_ub=bars_ub, x_start=x_phase2,
            k_list=k_list, time_per_k=float(args.time_per_k), verbose=args.verbose
        )
        x_best = lb.best_x.copy()
        lb_res = {
            "best_obj_patterns": int(lb.best_obj_patterns),
            "outside_cuts": int(lb.outside_cuts),
            "traces": lb.traces,
        }

    # choose final for outputs
    x_final = x_best.copy() if (args.mode in ("lb", "combined")) else x_phase2.copy()
    final_sum = summarize_solution(A, x_final, w, d, L)

    # legacy root keys (compat for older CSV collectors)
    legacy_bars = int(final_sum["total_bars"])
    legacy_patterns = int(final_sum["num_unique_patterns"])
    legacy_trim = float(final_sum["trim_loss_pct_demand"])
    legacy_realized = float(final_sum["trim_loss_pct_realized"])

    # stage summaries
    stage1_sum = {
        "stage": "stage1",
        "bars": int(stage1_bars),
        "patterns": int(stage1_patterns),
        "trim_loss_pct_demand": float(stage1_trim),
    }
    phase2_sum = summarize_solution(A, x_phase2, w, d, L) if phase2_res is not None else None

    # save solution.json (new + legacy)
    sol = {
        "mode": args.mode,
        "bars_ub": int(bars_ub),

        # new structured
        "stage1": stage1_sum,
        "phase2": phase2_res,
        "phase2_summary": phase2_sum,
        "lb": lb_res,
        "final": final_sum,

        # decision vectors
        "x_start": x_start.tolist(),
        "x_phase2": x_phase2.tolist(),
        "x_best": x_best.tolist(),

        # legacy flat keys for downstream scripts
        "bars": int(legacy_bars),
        "objective_patterns": int(legacy_patterns),
        "trim_loss_pct": float(legacy_trim),
        "realized_trim_loss_pct": float(legacy_realized),
    }
    save_json(sol, str(out_prefix) + "_solution.json")

    # save counts.json (THIS is what CSV should read)
    save_json(final_sum, str(out_prefix) + "_counts.json")

    # console
    print("完成：")
    print(f"  bars_ub = {bars_ub} (final bars = {final_sum['total_bars']})")
    print(f"  patterns: stage1={stage1_patterns} -> final={final_sum['num_unique_patterns']}")
    print(f"  trim_loss_pct_demand: {final_sum['trim_loss_pct_demand']:.4f}%")
    print(f"  trim_loss_pct_realized: {final_sum['trim_loss_pct_realized']:.4f}%")
    print(f"  输出: {str(out_prefix)}_counts.json / {str(out_prefix)}_solution.json")


if __name__ == "__main__":
    main()
