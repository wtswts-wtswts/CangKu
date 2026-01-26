# -*- coding: utf-8 -*-
"""
剪枝模块（详细中文注释）
（增加：模块级开关 + 统一配置入口 set_prune_config）
"""

from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np
import random

Candidate = Dict[str, Any]

# ===== 模块级开关（默认全部关闭，确保先跑通） =====
ENABLE_C3 = False      # 模式级支配
ENABLE_C2 = False      # 同根数支配
ENABLE_C1 = False      # 跨根数支配
ENABLE_M2 = False      # 历史单调性剪枝
ENABLE_GATE = False    # 启发式门控筛选（Top-K + 回退）

def set_prune_config(
    enable_c3: Optional[bool] = None,
    enable_c2: Optional[bool] = None,
    enable_c1: Optional[bool] = None,
    enable_m2: Optional[bool] = None,
    enable_gate: Optional[bool] = None,
):
    """
    统一设置剪枝/门控开关，可由外部调用（例如 run_txt_instance.py 在解析命令行后调用）
    """
    global ENABLE_C3, ENABLE_C2, ENABLE_C1, ENABLE_M2, ENABLE_GATE
    if enable_c3 is not None: ENABLE_C3 = bool(enable_c3)
    if enable_c2 is not None: ENABLE_C2 = bool(enable_c2)
    if enable_c1 is not None: ENABLE_C1 = bool(enable_c1)
    if enable_m2 is not None: ENABLE_M2 = bool(enable_m2)
    if enable_gate is not None: ENABLE_GATE = bool(enable_gate)


def _demand_mask(d_remain: np.ndarray) -> np.ndarray:
    return (np.asarray(d_remain, dtype=int) > 0).astype(np.int32)


def dominance_prune_modes_C3(candidates: List[Candidate]) -> List[Candidate]:
    """
    C3：模式级支配（删除被支配模式的所有动作）
    - 仅当 ENABLE_C3 为 True 时启用
    """
    if not ENABLE_C3 or len(candidates) <= 1:
        return candidates
    modes = {}
    for c in candidates:
        pid = int(c['pat_idx'])
        if pid not in modes:
            modes[pid] = {'a': c['a'], 'trim': int(c['trim'])}
    pids = list(modes.keys())
    dominated = set()
    for i in range(len(pids)):
        if pids[i] in dominated:
            continue
        Ai, trimA = modes[pids[i]]['a'], modes[pids[i]]['trim']
        for j in range(len(pids)):
            if i == j or pids[j] in dominated:
                continue
            Bj, trimB = modes[pids[j]]['a'], modes[pids[j]]['trim']
            if np.all(Ai >= Bj) and (trimA <= trimB):
                dominated.add(pids[j])
    if len(dominated) == 0:
        return candidates
    kept: List[Candidate] = [c for c in candidates if int(c['pat_idx']) not in dominated]
    return kept


def dominance_prune_within_candidates(
    candidates: List[Candidate],
    d_remain: np.ndarray,
    enable_C1: Optional[bool] = None,
    enable_C2: Optional[bool] = None
) -> List[Candidate]:
    """
    候选之间支配（安全、带需求掩码）：
    - C2（同根数）：若 k_A = k_B，且 cov(A)·M ≥ cov(B)·M（逐物品），trim_A ≤ trim_B，则删 B
    - C1（跨根数）：若 k_A ≤ k_B，且 cov(A)·M ≥ cov(B)·M（逐物品），waste(A) ≤ waste(B)，则删 B
    - 开关优先取入参；若入参为 None，使用模块级 ENABLE_C1/ENABLE_C2
    """
    _C1 = ENABLE_C1 if enable_C1 is None else bool(enable_C1)
    _C2 = ENABLE_C2 if enable_C2 is None else bool(enable_C2)
    if (not _C1 and not _C2) or len(candidates) <= 1:
        return candidates

    M = _demand_mask(d_remain)
    buckets: Dict[int, List[int]] = {}
    for idx, c in enumerate(candidates):
        buckets.setdefault(int(c['k']), []).append(idx)
    dominated = set()

    # C2：同根数
    if _C2:
        for k, idxs in buckets.items():
            n = len(idxs)
            for ii in range(n):
                i = idxs[ii]
                if i in dominated:
                    continue
                Ai = candidates[i]
                covA = Ai['a'] * int(Ai['k'])
                trimA = int(Ai['trim'])
                for jj in range(n):
                    if ii == jj:
                        continue
                    j = idxs[jj]
                    if j in dominated:
                        continue
                    Bj = candidates[j]
                    covB = Bj['a'] * int(Bj['k'])
                    trimB = int(Bj['trim'])
                    if np.all(covA[M == 1] >= covB[M == 1]) and (trimA <= trimB):
                        dominated.add(j)

    # C1：跨根数
    if _C1:
        all_idxs = list(range(len(candidates)))
        all_idxs.sort(key=lambda i: int(candidates[i]['k']))
        for ii in range(len(all_idxs)):
            i = all_idxs[ii]
            if i in dominated:
                continue
            Ai = candidates[i]
            kA = int(Ai['k'])
            covA = Ai['a'] * kA
            wasteA = kA * int(Ai['trim'])
            for jj in range(ii + 1, len(all_idxs)):
                j = all_idxs[jj]
                if j in dominated:
                    continue
                Bj = candidates[j]
                kB = int(Bj['k'])
                if kA > kB:
                    continue
                covB = Bj['a'] * kB
                wasteB = kB * int(Bj['trim'])
                if np.all(covA[M == 1] >= covB[M == 1]) and (wasteA <= wasteB):
                    dominated.add(j)

    kept = [candidates[i] for i in range(len(candidates)) if i not in dominated]
    return kept


def monotonicity_prune_M2_vs_history(candidates: List[Candidate], history_patterns: np.ndarray) -> List[Candidate]:
    """
    M2：对历史的单调性剪枝
    - 仅当 ENABLE_M2 为 True 时启用
    """
    if not ENABLE_M2 or len(candidates) == 0 or history_patterns is None or history_patterns.size == 0:
        return candidates
    H = history_patterns  # [m, |H|]
    kept: List[Candidate] = []
    for c in candidates:
        a = c['a']
        # trim 判据在缺少历史 trim 时保守不使用
        dominated = False
        for j in range(H.shape[1]):
            h = H[:, j]
            if np.all(a <= h):
                dominated = True
                break
        if not dominated:
            kept.append(c)
    return kept


def history_dedup(candidates: List[Candidate], history_actions: Set[Tuple[str, int]]) -> List[Candidate]:
    """
    历史去重：避免重复尝试完全相同的（模式，k）动作
    """
    kept: List[Candidate] = []
    for c in candidates:
        key = (hash(c['a'].tobytes()), int(c['k']))
        if key not in history_actions:
            kept.append(c)
    return kept


def heuristic_filter_with_fallback(
    candidates: List[Candidate],
    enable: Optional[bool] = None,
    K0: int = 64,
    R: int = 5,
    M: int = 16,
    state: Optional[Dict[str, Any]] = None
) -> List[Candidate]:
    """
    启发式阈值过滤（默认关闭）
    - 入参 enable 优先；None 时取模块级 ENABLE_GATE
    """
    use_gate = ENABLE_GATE if enable is None else bool(enable)
    if not use_gate or len(candidates) <= K0:
        return candidates

    def proxy_score(c: Candidate) -> float:
        frac = float(c.get('frac', 0.0))
        trim = float(c.get('trim', 0.0))
        return 1.0 * frac - 0.1 * trim

    ranked = sorted(candidates, key=proxy_score, reverse=True)
    kept = ranked[:K0]
    removed = ranked[K0:]

    if state is not None:
        cache = state.setdefault('cache', [])
        cache.extend(removed)
        R_cnt = int(state.get('no_improve_rounds', 0))
        if R_cnt >= R and len(cache) > 0:
            back = random.sample(cache, k=min(M, len(cache)))
            kept.extend(back)
            state['cache'] = [x for x in cache if x not in back]
    return kept