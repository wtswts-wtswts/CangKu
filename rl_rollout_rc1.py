# -*- coding: utf-8 -*-
"""
rl_rollout_rc.py  (FIXED: pending:contentReference[oaicite:2]{index=2})

关键修复：
1) pending 列即使 frac==0，也强制加入 RL 构图/排名集合，否则 rank_pass 永远 False；
2) pending 是“上一轮 ILP 输出”，只在下一轮 CG 收敛 + RL 打分后才尝试 commit；
3) 每步都会记录 pending_uid_in/pending_x_in，便于核对链路；
4) history_actions 用稳定 UID（sha1）而不是 Python hash()。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import numpy as np
import torch
import gurobipy as grb

from CG import CGRunner
from pattern_cache import PatternPool
import pruning as pr
from ilp_reuse import solve_epsilon_ilp


def _uid(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=np.int32).ravel()
    return hashlib.sha1(a.tobytes()).hexdigest()


def _adaptive_m_pct(d_remain: np.ndarray, d_init: np.ndarray, m_min: float, m_max: float, alpha: float) -> float:
    R0 = float(np.sum(d_init))
    if R0 <= 0:
        return float(m_min)
    Rt = float(np.sum(d_remain))
    ratio = max(0.0, min(1.0, Rt / R0))
    mt = m_min + (m_max - m_min) * (ratio ** float(alpha))
    return float(max(m_min, min(m_max, mt)))


def _remaining_total_length(w: np.ndarray, d: np.ndarray) -> int:
    return int(np.inner(w.astype(int), d.astype(int)))


def _solve_last_pattern_mip(w: np.ndarray, d_remain: np.ndarray, L: int, timelimit: float = 0.2) -> np.ndarray:
    w = np.asarray(w, dtype=int)
    d = np.asarray(d_remain, dtype=int).clip(min=0)
    m = len(w)
    if int(d.sum()) == 0:
        return np.zeros(m, dtype=int)

    model = grb.Model("final_bar_fill")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(max(0.05, timelimit))

    a = model.addMVar(shape=m, vtype=grb.GRB.INTEGER, lb=0, name="a")
    model.addConstr(a <= d)
    model.addConstr((w @ a) <= int(L))
    model.setObjective(w @ a, grb.GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        return np.zeros(m, dtype=int)
    sol = np.array(a.X, dtype=float).round().astype(int)
    sol = np.minimum(sol, d)
    if int(np.inner(w, sol)) > int(L):
        return np.zeros(m, dtype=int)
    return sol


def _ensure_pattern_in_hist_pool(hist_pool: PatternPool, a: np.ndarray) -> int:
    a = np.asarray(a, dtype=int).reshape(-1)
    hist_pool.try_add(a)
    P = hist_pool.to_numpy()
    for j in range(P.shape[1]):
        if np.array_equal(P[:, j].astype(int), a):
            return int(j)
    raise RuntimeError("pattern not found after insertion")


def _collect_fractional_candidates(P: np.ndarray, frac: np.ndarray, w: np.ndarray, L: int) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for p in range(P.shape[1]):
        fp = float(frac[p]) if p < len(frac) else 0.0
        if fp <= 1e-12:
            continue
        a = P[:, p].astype(int)
        used_len = int(np.inner(w, a))
        trim = int(L - used_len)

        if fp < 1.0 - 1e-12:
            cands.append({'pat_idx': p, 'a': a, 'trim': trim, 'used_len': used_len, 'frac': fp, 'k': 1, 'action': 'ceil'})
        else:
            k_floor = int(np.floor(fp))
            if k_floor > 0:
                cands.append({'pat_idx': p, 'a': a, 'trim': trim, 'used_len': used_len, 'frac': fp, 'k': k_floor, 'action': 'floor'})
            cands.append({'pat_idx': p, 'a': a, 'trim': trim, 'used_len': used_len, 'frac': fp, 'k': k_floor + 1, 'action': 'ceil'})
    return cands


def _choose_rounding_rule(cands_of_col: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(cands_of_col) == 1:
        out = dict(cands_of_col[0])
        out["k"] = max(1, int(out.get("k", 1)))
        return out

    fp = float(cands_of_col[0].get("frac", 0.0))
    frac_part = fp - np.floor(fp)

    floor_c = next((c for c in cands_of_col if c.get("action") == "floor"), None)
    ceil_c = next((c for c in cands_of_col if c.get("action") == "ceil"), None)

    chosen = ceil_c if frac_part >= 0.5 else floor_c
    if chosen is None:
        chosen = cands_of_col[0]
    chosen = dict(chosen)
    chosen["k"] = max(1, int(chosen.get("k", 1)))
    return chosen


def _build_graph_for_policy(
    w: np.ndarray,
    L: int,
    d_remain: np.ndarray,
    P: np.ndarray,
    frac: np.ndarray,
    last_pat_idx: Optional[int],
    pending_pat_idx: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    ✅ 修复点：cols = {p | frac[p]>0} ∪ {pending_pat_idx}
    返回 cols：图中第 j 列对应 P 的列索引。
    """
    m, n = P.shape[0], P.shape[1]
    item_feats = np.stack([w / float(L), d_remain.astype(np.float32)], axis=1).astype(np.float32)

    cols: List[int] = []
    for p in range(n):
        fp = float(frac[p]) if p < len(frac) else 0.0
        if fp > 1e-12:
            cols.append(p)
    if pending_pat_idx is not None and pending_pat_idx not in cols:
        cols.append(int(pending_pat_idx))

    if len(cols) == 0:
        return (torch.zeros((m, 2), dtype=torch.float32),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.float32),
                torch.zeros((0, 4), dtype=torch.float32),
                [])

    edge_i, edge_j, edge_feat = [], [], []
    col_feats = []

    for j_local, p in enumerate(cols):
        a = P[:, p].astype(int)
        used_len = int(np.inner(w, a))
        trim = int(L - used_len)
        same = 1.0 if (last_pat_idx is not None and p == last_pat_idx) else 0.0
        fp = float(frac[p]) if p < len(frac) else 0.0

        col_feats.append([fp, float(trim) / float(L), same, 1.0 - float(trim) / float(L)])

        for i in range(m):
            if a[i] > 0:
                edge_i.append(i)
                edge_j.append(j_local)
                edge_feat.append([float(a[i])])

    item_t = torch.from_numpy(item_feats)
    edge_index_t = torch.tensor([edge_i, edge_j], dtype=torch.long)
    edge_feat_t = torch.tensor(edge_feat, dtype=torch.float32)
    col_feat_t = torch.tensor(col_feats, dtype=torch.float32)
    return item_t, edge_index_t, edge_feat_t, col_feat_t, cols


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
):
    w = np.asarray(w, dtype=int)
    d_init = np.asarray(d_init, dtype=int)
    d_remain = d_init.copy()

    cg = CGRunner(w=w, L=L, use_pricer_cache=use_pricer_cache, pricer_seed=pricer_seed)
    P = cg.build_initial_pool(d_remain=d_remain)
    hist_pool = PatternPool.from_P(P)

    sequence: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {"ilp_traces": []}

    last_pat_idx: Optional[int] = None

    pending_uid: Optional[str] = None
    pending_x: int = 0

    history_actions = set()

    step_no = 0
    while np.any(d_remain > 0):
        step_no += 1

        # 终止：剩余总长度 <= L，用 final_mip
        if _remaining_total_length(w, d_remain) <= int(L):
            a_last = _solve_last_pattern_mip(w, d_remain, L, timelimit=max(0.05, timelimit))
            if int(a_last.sum()) > 0:
                pat_idx = _ensure_pattern_in_hist_pool(hist_pool, a_last)
                sequence.append({
                    "step": step_no,
                    "pat_idx": pat_idx,
                    "a": a_last.astype(int).tolist(),
                    "k": 1,
                    "action": "final_mip",
                    "is_reuse": False,
                    "pending_uid_in": pending_uid,
                    "pending_x_in": int(pending_x),
                })
                d_remain = np.maximum(0, d_remain - a_last.astype(int))
            break

        # 1) CG 收敛
        P, pi, frac, added_cols = cg.run_until_no_neg_rc(
            P,
            d_remain,
            gate_alpha=None,
            solver=solver,
            timelimit=timelimit,
            pricing_topk=1,
            pricing_noise=0.0,
            max_cols=0,
            recent_keep=200,
            x_eps=1e-9,
            rng=None,
        )

        # 更新历史池
        for add in added_cols:
            hist_pool.try_add(add["a"].astype(int))

        # 找 pending 在 P 里的列索引（可能不存在）
        pending_pat_idx: Optional[int] = None
        if pending_uid is not None and pending_x > 0:
            for p in range(P.shape[1]):
                if _uid(P[:, p]) == pending_uid:
                    pending_pat_idx = int(p)
                    break

        # 2) RL 打分：frac>0 + pending 强制加入
        item_t, edge_idx_t, edge_feat_t, col_feat_t, cols = _build_graph_for_policy(
            w, L, d_remain, P, frac, last_pat_idx, pending_pat_idx
        )

        probs_np = None
        if len(cols) > 0:
            with torch.no_grad():
                logits = policy.action_logits(item_t, edge_idx_t, edge_feat_t, col_feat_t)
            # logits: [n_cols_graph]
            probs = torch.softmax(logits, dim=0)
            probs_np = probs.detach().cpu().numpy().reshape(-1)

        # 计算 top-m%
        if reuse_m_adaptive:
            m_pct = _adaptive_m_pct(d_remain, d_init, reuse_m_min, reuse_m_max, reuse_m_alpha)
        else:
            m_pct = float(reuse_top_pct)

        top_set_cols = set()
        topN = 1
        if probs_np is not None and len(probs_np) > 0:
            order = np.argsort(-probs_np)
            topN = max(1, int(np.ceil(float(m_pct) * len(order))))
            top_set_cols = set([int(cols[g]) for g in order[:topN].tolist()])

        # 3) ✅ 先尝试 commit pending
        committed = False
        chosen: Dict[str, Any] = {}

        if pending_pat_idx is not None and pending_x > 0:
            a_pending = P[:, pending_pat_idx].astype(int)
            # reduced cost = 1 - pi·a
            rc_val = 1.0 - float(np.dot(np.asarray(pi, dtype=float), a_pending.astype(float)))
            rc_pass = (rc_val <= 0.0 + float(rc_eps))
            rank_pass = (pending_pat_idx in top_set_cols)

            # feasible k
            k_feas = 10**9
            for i in range(len(d_remain)):
                if a_pending[i] > 0:
                    k_feas = min(k_feas, int(d_remain[i] // a_pending[i]))
            if k_feas == 10**9:
                k_feas = 0
            k_use = int(min(int(pending_x), int(k_feas)))

            if rc_pass and rank_pass and k_use > 0:
                chosen = {
                    "step": step_no,
                    "pat_idx": int(pending_pat_idx),
                    "a": a_pending.tolist(),
                    "k": int(k_use),
                    "action": "commit_reuse",
                    "is_reuse": True,
                    "reduced_cost": float(rc_val),
                    "m_pct": float(m_pct),
                    "topN": int(topN),
                    "pending_uid_in": pending_uid,
                    "pending_x_in": int(pending_x),
                }
                committed = True

        # 4) 不 commit 才走 normal RL 选择 + rounding
        if not committed:
            cands = _collect_fractional_candidates(P, frac, w=w, L=L)
            cands = pr.dominance_prune_modes_C3(cands)
            cands = pr.history_dedup(cands, history_actions)
            if enable_heuristic_topk:
                cands = pr.heuristic_filter_with_fallback(cands, enable=True, K0=topk_K0)

            if probs_np is None or len(cols) == 0:
                chosen = dict(cands[0])
            else:
                order = np.argsort(-probs_np)
                sel_local = int(order[0])
                chosen_pat_idx = int(cols[sel_local])
                cands_of_col = [c for c in cands if int(c.get("pat_idx")) == chosen_pat_idx]
                chosen = _choose_rounding_rule(cands_of_col) if len(cands_of_col) else dict(cands[0])

            chosen = dict(chosen)
            chosen.update({
                "step": step_no,
                "is_reuse": False,
                "m_pct": float(m_pct),
                "topN": int(topN),
                "pending_uid_in": pending_uid,
                "pending_x_in": int(pending_x),
            })

        # 执行动作
        pat_idx = _ensure_pattern_in_hist_pool(hist_pool, np.asarray(chosen["a"], dtype=int))
        chosen["pat_idx"] = int(pat_idx)

        sequence.append(chosen)

        d_remain = np.maximum(0, d_remain - np.asarray(chosen["a"], dtype=int) * int(chosen["k"]))
        history_actions.add((_uid(np.asarray(chosen["a"], dtype=int)), int(chosen["k"])))
        last_pat_idx = int(pat_idx)

        # 5) 本步结束后：跑 ILP 产出 next pending（用于下一步）
        next_pending_uid, next_pending_x = None, 0
        if ilp_enable and probs_np is not None and len(cols) > 0:
            m_items = int(P.shape[0])

            # auto topk/hist if <=0
            if int(ilp_topk) <= 0:
                ilp_topk_eff = int(max(ilp_topk_min, min(ilp_topk_max, int(np.ceil(float(ilp_topk_mult) * m_items)))))
            else:
                ilp_topk_eff = int(ilp_topk)

            if int(ilp_hist) <= 0:
                ilp_hist_eff = int(max(ilp_hist_min, min(ilp_hist_max, int(np.ceil(float(ilp_hist_mult) * m_items)))))
            else:
                ilp_hist_eff = int(ilp_hist)

            order = np.argsort(-probs_np)
            topk_local = order[: min(ilp_topk_eff, len(order))].tolist()
            topk_cols = [int(cols[g]) for g in topk_local]

            # hist: recent unique uids
            hist_uids: List[str] = []
            for st in reversed(sequence):
                u = _uid(np.asarray(st.get("a", []), dtype=int))
                if u not in hist_uids:
                    hist_uids.append(u)
                if len(hist_uids) >= ilp_hist_eff:
                    break

            hist_cols: List[int] = []
            if len(hist_uids) > 0:
                for p in range(P.shape[1]):
                    if _uid(P[:, p]) in hist_uids:
                        hist_cols.append(int(p))

            small_cols = list(dict.fromkeys(topk_cols + hist_cols))
            if len(small_cols) > int(ilp_max_cols):
                small_cols = small_cols[: int(ilp_max_cols)]

            A_small = P[:, small_cols].astype(int)
            uids_small = [_uid(P[:, p]) for p in small_cols]

            bars_lb = float(np.sum(np.asarray(frac, dtype=float)))
            B0 = int(np.ceil(bars_lb))

            used_delta = int(ilp_delta)
            relax_tries = 0
            res = None
            while True:
                res = solve_epsilon_ilp(
                    A=A_small,
                    d=d_remain,
                    B0=B0,
                    delta=int(used_delta),
                    timelimit=float(ilp_timelimit),
                    pattern_uids=uids_small,
                    verbose=False,
                )
                status = (res or {}).get("status", "unknown")
                if status != "infeasible" or not ilp_relax_on_infeasible:
                    break
                relax_tries += 1
                if relax_tries > int(ilp_relax_max):
                    break
                used_delta += 1

            info["ilp_traces"].append({
                "step": int(step_no),
                "B0": int(B0),
                "delta": int(used_delta),
                "status": str((res or {}).get("status")),
                "small_cols": int(len(small_cols)),
                "best_uid": (res or {}).get("best_uid"),
                "best_x": int((res or {}).get("best_x", 0)),
            })

            if res is not None and res.get("best_uid") is not None and int(res.get("best_x", 0)) > 0:
                next_pending_uid = str(res["best_uid"])
                next_pending_x = int(res["best_x"])

        pending_uid, pending_x = next_pending_uid, int(next_pending_x)

    return hist_pool.to_numpy().astype(int), sequence, info
