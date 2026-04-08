# -*- coding: utf-8 -*-
"""RL rollout with mini-pool window replacement and global-demand-consistent bookkeeping."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

from CG1 import CGRunner
from pattern_cache1 import PatternPool


def _uid(a: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(a, dtype=np.int32).ravel().tobytes()).hexdigest()


def _remaining_total_length(w: np.ndarray, d: np.ndarray) -> int:
    return int(np.inner(w.astype(int), d.astype(int)))


def _solve_last_pattern_mip(w: np.ndarray, d_remain: np.ndarray, L: int, timelimit: float = 0.2) -> np.ndarray:
    w = np.asarray(w, dtype=int)
    d = np.asarray(d_remain, dtype=int).clip(min=0)
    if int(d.sum()) == 0:
        return np.zeros_like(d)

    model = gp.Model("final_bar_fill")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(max(0.05, timelimit))

    a = model.addMVar(shape=len(w), vtype=GRB.INTEGER, lb=0, name="a")
    model.addConstr(a <= d)
    model.addConstr((w @ a) <= int(L))
    model.setObjective(w @ a, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return np.zeros_like(d)
    sol = np.array(a.X, dtype=float).round().astype(int)
    return np.minimum(sol, d)


def _ensure_pattern_in_hist_pool(hist_pool: PatternPool, a: np.ndarray) -> int:
    hist_pool.try_add(np.asarray(a, dtype=int).reshape(-1))
    P = hist_pool.to_numpy()
    for j in range(P.shape[1]):
        if np.array_equal(P[:, j].astype(int), np.asarray(a, dtype=int).reshape(-1)):
            return int(j)
    raise RuntimeError("pattern not found after insertion")


def _build_graph_for_policy(
    w: np.ndarray,
    L: int,
    d_remain: np.ndarray,
    P: np.ndarray,
    frac: np.ndarray,
    last_pat_idx: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    m, n = P.shape
    item_feats = np.stack([w / float(L), d_remain.astype(np.float32)], axis=1).astype(np.float32)

    cols = [p for p in range(n) if float(frac[p]) > 1e-12]
    if len(cols) == 0:
        return (torch.zeros((m, 2), dtype=torch.float32),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.float32),
                torch.zeros((0, 4), dtype=torch.float32),
                [])

    edge_i, edge_j, edge_feat, col_feats = [], [], [], []
    for j_local, p in enumerate(cols):
        a = P[:, p].astype(int)
        used_len = int(np.inner(w, a))
        trim = int(L - used_len)
        same = 1.0 if (last_pat_idx is not None and p == last_pat_idx) else 0.0
        fp = float(frac[p])
        col_feats.append([fp, float(trim) / float(L), same, 1.0 - float(trim) / float(L)])
        for i in range(m):
            if a[i] > 0:
                edge_i.append(i)
                edge_j.append(j_local)
                edge_feat.append([float(a[i])])

    return (
        torch.from_numpy(item_feats),
        torch.tensor([edge_i, edge_j], dtype=torch.long),
        torch.tensor(edge_feat, dtype=torch.float32),
        torch.tensor(col_feats, dtype=torch.float32),
        cols,
    )


def _solve_mini_pool_replacement(
    P_pool: np.ndarray,
    d_window: np.ndarray,
    bars_exec_window: int,
    timelimit: float,
) -> Dict[str, Any]:
    if int(d_window.sum()) == 0:
        return {"status": "optimal", "x": np.zeros(P_pool.shape[1], dtype=int)}

    model = gp.Model("mini_pool_replace")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(max(0.05, timelimit))

    m, n = P_pool.shape
    x = model.addMVar(shape=n, vtype=GRB.INTEGER, lb=0, name="x")
    y = model.addMVar(shape=n, vtype=GRB.BINARY, name="y")

    model.addConstr(P_pool @ x >= d_window, name="cover")
    model.addConstr(x.sum() <= int(max(0, bars_exec_window)), name="bars_cap")
    M = int(max(1, bars_exec_window))
    model.addConstr(x <= M * y, name="link")
    model.setObjective(y.sum(), GRB.MINIMIZE)
    model.optimize()

    st = int(model.Status)
    if st == GRB.INFEASIBLE:
        return {"status": "infeasible", "x": None}
    if st not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or model.SolCount <= 0:
        return {"status": f"error_{st}", "x": None}

    x_sol = np.array(x.X, dtype=float).round().astype(int)
    return {"status": "optimal" if st == GRB.OPTIMAL else "time_limit", "x": x_sol}


def rollout_generate_plan(
    w: np.ndarray,
    d_init: np.ndarray,
    L: int,
    policy,
    mode: str = "eval",
    solver: str = "gurobi",
    timelimit: float = 0.2,
    enable_heuristic_topk: bool = False,
    topk_K0: int = 64,
    use_pricer_cache: bool = False,
    pricer_seed: Optional[int] = None,
    reuse_m_adaptive: bool = True,
    reuse_top_pct: float = 0.10,
    reuse_m_min: float = 0.05,
    reuse_m_max: float = 0.25,
    reuse_m_alpha: float = 1.0,
    rc_eps: float = 1e-9,
    ilp_enable: bool = False,
    ilp_delta: int = 0,
    ilp_timelimit: float = 0.5,
    ilp_max_cols: int = 250,
    ilp_topk: int = 64,
    ilp_topk_mult: float = 3.0,
    ilp_topk_min: int = 32,
    ilp_topk_max: int = 128,
    ilp_hist: int = 64,
    ilp_hist_mult: float = 3.0,
    ilp_hist_min: int = 32,
    ilp_hist_max: int = 128,
    ilp_relax_on_infeasible: bool = False,
    ilp_relax_max: int = 2,
    rl_topk: int = 3,
    util_min_aux: float = 0.8,
    mini_pool_cap: int = 10,
    mini_pool_max_steps: int = 5,
    mini_ilp_timelimit: float = 0.5,
    tail_drop_last: int = 1,
    tail_util_threshold: float = 0.6,
    log_trace: bool = False,
    trace_json: str = "",
):
    w = np.asarray(w, dtype=int)
    d_init = np.asarray(d_init, dtype=int)

    cg = CGRunner(w=w, L=L, use_pricer_cache=use_pricer_cache, pricer_seed=pricer_seed)
    P = cg.build_initial_pool(d_remain=d_init)
    hist_pool = PatternPool.from_P(P)

    sequence: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {
        "ilp_traces": [],
        "window_traces": traces,
        "cnt_pool_solve_ok": 0,
        "cnt_pool_solve_fail": 0,
        "cnt_tail_drop_last": 0,
    }

    d_remain = d_init.copy()
    d_before_window = d_remain.copy()
    d_tmp = d_remain.copy()
    prod_exec_total = np.zeros_like(d_init)
    steps_in_window = 0
    bars_exec_window = 0
    window_pool: List[np.ndarray] = []
    last_pat_idx: Optional[int] = None
    step_no = 0

    def flush_window(reason: str):
        nonlocal d_remain, d_before_window, d_tmp, prod_exec_total, steps_in_window, bars_exec_window, window_pool
        if steps_in_window <= 0:
            return
        P_pool = np.stack(window_pool, axis=1).astype(int)
        d_window = np.minimum(d_before_window, prod_exec_total)
        res = _solve_mini_pool_replacement(P_pool, d_window, bars_exec_window, mini_ilp_timelimit)

        solve_status = str(res.get("status", "error"))
        x_sol = res.get("x", None)
        if x_sol is None:
            solve_status = "infeasible"
            x_sol = np.ones(P_pool.shape[1], dtype=int)
            x_sol[:] = 0
            for p in range(P_pool.shape[1]):
                x_sol[p] = 1
                if int(np.sum(P_pool @ x_sol >= d_window)):
                    break
            info["cnt_pool_solve_fail"] += 1
            prod_pool_total = prod_exec_total.copy()
            bars_pool = int(bars_exec_window)
            pat_pool = int(len({ _uid(c) for c in window_pool }))
        else:
            info["cnt_pool_solve_ok"] += 1
            prod_pool_total = (P_pool @ np.asarray(x_sol, dtype=int)).astype(int)
            bars_pool = int(np.sum(x_sol))
            pat_pool = int(np.sum(np.asarray(x_sol, dtype=int) > 0))

        dropped = False
        util_last = None
        if int(tail_drop_last) == 1 and bars_pool > 0 and x_sol is not None:
            used = np.where(np.asarray(x_sol, dtype=int) > 0)[0]
            if len(used) > 0:
                utils = [float(np.dot(w, P_pool[:, j])) / float(L) for j in used]
                j_drop = int(used[int(np.argmin(utils))])
                util_last = float(np.min(utils))
                if util_last < float(tail_util_threshold) and x_sol[j_drop] > 0:
                    x_sol[j_drop] -= 1
                    prod_pool_total = (P_pool @ np.asarray(x_sol, dtype=int)).astype(int)
                    bars_pool = int(np.sum(x_sol))
                    pat_pool = int(np.sum(np.asarray(x_sol, dtype=int) > 0))
                    dropped = True
                    info["cnt_tail_drop_last"] += 1

        covered = np.minimum(d_before_window, prod_pool_total)
        d_after_window = (d_before_window - covered).astype(int)
        overprod = np.maximum(0, prod_pool_total - d_before_window)

        tr = {
            "trigger": reason,
            "pool_size": int(len(window_pool)),
            "steps_in_window": int(steps_in_window),
            "solve_status": solve_status,
            "bars_exec_window": int(bars_exec_window),
            "bars_pool_solution": int(bars_pool),
            "patterns_exec_window": int(len({_uid(c) for c in window_pool})),
            "patterns_pool_solution": int(pat_pool),
            "tail_drop_last": bool(dropped),
            "util_last": util_last,
            "sum_d_before_window": int(np.sum(d_before_window)),
            "sum_d_after_window": int(np.sum(d_after_window)),
            "overprod_len": float(np.dot(w.astype(float), overprod.astype(float))),
        }
        traces.append(tr)
        if log_trace:
            print(f"[window] trigger={reason} pool={tr['pool_size']} steps={tr['steps_in_window']} status={solve_status} bars {bars_exec_window}->{bars_pool}")

        d_remain = d_after_window
        d_before_window = d_after_window.copy()
        d_tmp = d_after_window.copy()
        prod_exec_total = np.zeros_like(d_init)
        steps_in_window = 0
        bars_exec_window = 0
        window_pool = []

    while np.any(d_remain > 0):
        step_no += 1

        if _remaining_total_length(w, d_tmp) <= int(L):
            a_last = _solve_last_pattern_mip(w, d_tmp, L, timelimit=max(0.05, timelimit))
            if int(a_last.sum()) > 0:
                pat_idx = _ensure_pattern_in_hist_pool(hist_pool, a_last)
                sequence.append({"step": step_no, "pat_idx": pat_idx, "a": a_last.tolist(), "k": 1, "action": "final_mip"})
                d_tmp = np.maximum(0, d_tmp - a_last)
                prod_exec_total += a_last
                bars_exec_window += 1
                window_pool.append(a_last.copy())
                steps_in_window += 1
            flush_window("max_steps")
            break

        P, pi, frac, added_cols = cg.run_until_no_neg_rc(
            P, d_tmp, gate_alpha=None, solver=solver, timelimit=timelimit,
            pricing_topk=1, pricing_noise=0.0, max_cols=0, recent_keep=200, x_eps=1e-9, rng=None,
        )
        for add in added_cols:
            hist_pool.try_add(add["a"].astype(int))

        item_t, edge_idx_t, edge_feat_t, col_feat_t, cols = _build_graph_for_policy(w, L, d_tmp, P, frac, last_pat_idx)
        if len(cols) == 0:
            break

        with torch.no_grad():
            logits = policy.action_logits(item_t, edge_idx_t, edge_feat_t, col_feat_t)
        probs_np = torch.softmax(logits, dim=0).detach().cpu().numpy().reshape(-1)
        order = np.argsort(-probs_np)
        topk_local = order[: min(max(1, int(rl_topk)), len(order))]
        topk_cols = [int(cols[g]) for g in topk_local]

        top1_col = topk_cols[0]
        a_exec = P[:, top1_col].astype(int)
        pat_idx = _ensure_pattern_in_hist_pool(hist_pool, a_exec)
        last_pat_idx = int(top1_col)

        sequence.append({
            "step": step_no,
            "pat_idx": pat_idx,
            "a": a_exec.tolist(),
            "k": 1,
            "action": "top1_commit",
            "is_reuse": False,
        })

        d_tmp = np.maximum(0, d_tmp - a_exec)
        prod_exec_total += a_exec
        bars_exec_window += 1
        steps_in_window += 1

        window_pool.append(a_exec.copy())
        for c in topk_cols[1:]:
            a_aux = P[:, c].astype(int)
            util = float(np.dot(w, a_aux)) / float(L)
            if util >= float(util_min_aux):
                window_pool.append(a_aux.copy())

        reason = None
        if len(window_pool) >= int(mini_pool_cap):
            reason = "cap"
        elif steps_in_window >= int(mini_pool_max_steps):
            reason = "max_steps"
        elif not np.any(d_tmp > 0):
            reason = "done"
        if reason is not None:
            flush_window(reason)

    if trace_json:
        import json
        with open(trace_json, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

    return hist_pool.to_numpy().astype(int), sequence, info
