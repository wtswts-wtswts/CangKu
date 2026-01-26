# -*- coding: utf-8 -*-
"""
第一阶段 rollout（增强版：支持多列定价 + 列池管理）
做了什么：
- 透传 multi-column pricing 参数：pricing_topk / pricing_noise
- 透传 column management 参数：max_cols / recent_keep / x_eps
- 其余逻辑尽量保持不变
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import gurobipy as grb

from CG import CGRunner
from pattern_cache import PatternPool
import pruning as pr


def _remaining_total_length(w: np.ndarray, d_remain: np.ndarray) -> int:
    return int(np.inner(w.astype(int), d_remain.astype(int)))


def _solve_last_pattern_mip(w: np.ndarray, d_remain: np.ndarray, L: int, timelimit: float = 0.2) -> np.ndarray:
    w = np.asarray(w, dtype=int)
    d = np.asarray(d_remain, dtype=int).clip(min=0)
    m = len(w)

    if int(d.sum()) == 0:
        return np.zeros(m, dtype=int)

    model = grb.Model("final_bar_fill")
    model.Params.OutputFlag = 0
    if timelimit is not None:
        model.Params.TimeLimit = float(timelimit)

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


def _build_graph_for_policy(
    w: np.ndarray,
    L: int,
    d_remain: np.ndarray,
    P: np.ndarray,
    frac: np.ndarray,
    last_pat_idx: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    m, n = P.shape[0], P.shape[1]
    item_feats = np.stack([w / float(L), d_remain.astype(np.float32)], axis=1).astype(np.float32)

    edge_idx_i, edge_idx_j, edge_feat = [], [], []
    col_feats = []
    cols = []

    for p in range(n):
        a = P[:, p]
        if np.sum(a) == 0:
            continue
        fp = float(frac[p]) if p < len(frac) else 0.0
        if fp <= 1e-12:
            continue

        used_len = int(np.inner(w, a))
        trim = int(L - used_len)
        same = 1.0 if (last_pat_idx is not None and p == last_pat_idx) else 0.0
        col_feats.append([fp, float(trim) / float(L), same, 1.0 - float(trim) / float(L)])
        cols.append(p)

        col_local = len(cols) - 1
        for i in range(m):
            if a[i] > 0:
                edge_idx_i.append(i)
                edge_idx_j.append(col_local)
                edge_feat.append([float(a[i])])

    if len(cols) == 0:
        return (torch.zeros((m, 2), dtype=torch.float32),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.float32),
                torch.zeros((0, 4), dtype=torch.float32),
                [])

    item_t = torch.from_numpy(item_feats)
    edge_index_t = torch.tensor([edge_idx_i, edge_idx_j], dtype=torch.long)
    edge_feat_t = torch.tensor(edge_feat, dtype=torch.float32)
    col_feat_t = torch.tensor(col_feats, dtype=torch.float32)
    return item_t, edge_index_t, edge_feat_t, col_feat_t, cols


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
        return cands_of_col[0]
    fp = float(cands_of_col[0].get('frac', 0.0))
    frac_part = fp - np.floor(fp)

    floor_c = None
    ceil_c = None
    for c in cands_of_col:
        if c['action'] == 'floor':
            floor_c = c
        elif c['action'] == 'ceil':
            ceil_c = c
    if floor_c is None:
        return ceil_c
    if ceil_c is None:
        return floor_c
    return floor_c if frac_part < 0.5 else ceil_c


def rollout_generate_plan(
    w: np.ndarray,
    d_init: np.ndarray,
    L: int,
    policy,
    mode: str = 'eval',
    solver: str = 'gurobi',
    timelimit: float = 0.2,
    enable_gate: bool = False,
    enable_heuristic_topk: bool = False,
    topk_K0: int = 64,
    max_outer_steps: int = 100000,
    use_pricer_cache: bool = False,
    pricer_seed: Optional[int] = None,
    # === 新增：多列定价与列池管理参数 ===
    pricing_topk: int = 1,
    pricing_noise: float = 0.0,
    max_cols: int = 0,
    recent_keep: int = 200,
    x_eps: float = 1e-9,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    w = np.asarray(w, dtype=int)
    d_init = np.asarray(d_init, dtype=int)
    d_remain = d_init.copy()

    cg = CGRunner(w=w, L=L, use_pricer_cache=use_pricer_cache, pricer_seed=pricer_seed)

    P = cg.build_initial_pool(d_remain=d_remain)

    hist_pool = PatternPool.from_P(P)

    selected_sequence: List[Dict[str, Any]] = []
    history_actions: set = set()
    info: Dict[str, Any] = {
        'lp_traces': [],
        'added_cols_traces': [],
        'd_remain_traces': [],
        'step_stats_traces': [],
        'use_pricer_cache': bool(use_pricer_cache),
        'pricer_seed': pricer_seed,
        'pricing_topk': int(pricing_topk),
        'pricing_noise': float(pricing_noise),
        'max_cols': int(max_cols),
        'recent_keep': int(recent_keep),
        'x_eps': float(x_eps),
        'seed': seed,
    }
    last_pat_idx: Optional[int] = None

    rng = np.random.default_rng(seed if seed is not None else 0)

    for _step in range(int(max_outer_steps)):
        if not np.any(d_remain > 0):
            break

        if _remaining_total_length(w, d_remain) <= int(L):
            a_last = _solve_last_pattern_mip(w=w, d_remain=d_remain, L=L, timelimit=max(0.05, timelimit))
            if int(a_last.sum()) > 0:
                used_len = int(np.inner(w, a_last))
                trim_last = int(L - used_len)
                selected_sequence.append({
                    'pat_idx': -1,
                    'action': 'final_mip',
                    'k': 1,
                    'used_len': used_len,
                    'trim': trim_last,
                    'switched': int(last_pat_idx is not None),
                    'a': a_last.copy()
                })
                d_remain = np.maximum(d_remain - a_last, 0)
            break

        info['d_remain_traces'].append(d_remain.copy())

        gate_alpha = None
        if enable_gate:
            item_feats = np.stack([w / float(L), d_remain.astype(np.float32)], axis=1).astype(np.float32)
            item_feats_t = torch.from_numpy(item_feats)
            with torch.no_grad():
                gate_alpha_t = policy.pricing_mask(item_feats_t)
            gate_alpha = gate_alpha_t.detach().cpu().numpy()

        P, pi, frac, added_cols = cg.run_until_no_neg_rc(
            P,
            d_remain,
            gate_alpha=gate_alpha,
            solver=solver,
            timelimit=timelimit,
            pricing_topk=pricing_topk,
            pricing_noise=pricing_noise,
            max_cols=max_cols,
            recent_keep=recent_keep,
            x_eps=x_eps,
            rng=rng
        )

        for add in added_cols:
            a_new = add['a'].astype(int)
            hist_pool.try_add(a_new)

        info['lp_traces'].append({'pi': pi, 'frac': frac})
        info['added_cols_traces'].append(added_cols)

        num_cols = int(P.shape[1])
        num_frac_pos = int(np.sum(np.asarray(frac) > 1e-12)) if frac is not None else 0

        cands = _collect_fractional_candidates(P, frac, w=w, L=L)
        num_cands_before = int(len(cands))

        cands = pr.dominance_prune_modes_C3(cands)
        cands = pr.dominance_prune_within_candidates(cands, d_remain, enable_C1=None, enable_C2=None)
        # M2 仍然建议不要开（已证实会炸质量）；保留接口但默认关闭
        cands = pr.monotonicity_prune_M2_vs_history(cands, hist_pool.to_numpy())
        cands = pr.history_dedup(cands, history_actions)
        if enable_heuristic_topk:
            cands = pr.heuristic_filter_with_fallback(cands, enable=None, K0=topk_K0)

        num_cands_after = int(len(cands))
        info['step_stats_traces'].append({
            'num_cols': num_cols,
            'num_frac_pos': num_frac_pos,
            'num_cands_before': num_cands_before,
            'num_cands_after': num_cands_after
        })

        if len(cands) == 0:
            best = None
            best_gain = -1
            for p in range(P.shape[1]):
                a = P[:, p].astype(int)
                if int(a.sum()) == 0:
                    continue
                gain = int(np.inner(w, np.minimum(d_remain, a)))
                if gain > best_gain:
                    best_gain = gain
                    best = (p, a)
            if best is None or best_gain <= 0:
                info['stuck'] = True
                break
            p, a = best
            chosen = {'pat_idx': int(p), 'a': a, 'frac': 0.0, 'k': 1, 'action': 'fallback'}
        else:
            item_t, edge_idx_t, edge_feat_t, col_feat_t, col_index_map = _build_graph_for_policy(
                w, L, d_remain, P, frac, last_pat_idx
            )
            if len(col_index_map) == 0:
                chosen = cands[0]
            else:
                with torch.no_grad():
                    logits = policy.action_logits(item_t, edge_idx_t, edge_feat_t, col_feat_t)
                col_id_in_graph = policy.choose_action(logits, mode=mode)
                chosen_pat_idx = int(col_index_map[col_id_in_graph])

                cands_of_col = [c for c in cands if int(c['pat_idx']) == chosen_pat_idx]
                chosen = _choose_rounding_rule(cands_of_col) if len(cands_of_col) else cands[0]

        k = int(chosen['k'])
        a = chosen['a'].astype(int)
        d_remain = np.maximum(d_remain - k * a, 0)

        used_len = int(np.inner(w, a))
        trim = int(L - used_len)
        if trim < 0:
            info.setdefault('negative_trim_events', []).append({
                'step': int(_step),
                'pat_idx': int(chosen.get('pat_idx', -1)),
                'used_len': used_len,
                'L': int(L)
            })
            trim = max(trim, 0)

        switched = int(last_pat_idx is not None and (int(chosen.get('pat_idx', -1)) != int(last_pat_idx)) and (k > 0))
        selected_sequence.append({
            'pat_idx': int(chosen.get('pat_idx', -1)),
            'action': str(chosen.get('action', '')),
            'k': k,
            'used_len': used_len,
            'trim': trim,
            'switched': switched,
            'a': a.copy()
        })

        history_actions.add((hash(a.tobytes()), k))
        if k > 0 and int(chosen.get('pat_idx', -1)) >= 0:
            last_pat_idx = int(chosen['pat_idx'])

    return hist_pool.to_numpy(), selected_sequence, info