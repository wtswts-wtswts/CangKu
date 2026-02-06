# -*- coding: utf-8 -*-
"""
pruning.py  (兼容版：接口齐全，但只真正启用 C3)

目标：
- 你的 rl_rollout_rc.py 可能调用一串 pruning 函数（C1/C2/M2/GATE/heuristic 等），
  即使你想删掉其它策略，也必须“留函数入口”避免 AttributeError。
- 这里的策略实现原则：
  - 只真正启用 C3（模式级去重/保守支配），默认开启
  - 其它策略函数都保留同名接口，但默认做 no-op（返回原 candidates）
- 稳定 UID：
  - pat_uid(a) 使用 sha1(a.tobytes())，跨进程稳定
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import numpy as np

# =========================
# Global switches
# =========================
ENABLE_C3: bool = True  # 你要求：只保留 C3 且默认开启

# 其余开关保留（兼容旧代码），但默认无效
ENABLE_C1: bool = False
ENABLE_C2: bool = False
ENABLE_M2: bool = False
ENABLE_GATE: bool = False


# =========================
# Stable pattern UID
# =========================
def pat_uid(a: np.ndarray) -> str:
    """Stable UID for a pattern vector a."""
    a = np.asarray(a, dtype=np.int32).ravel()
    return hashlib.sha1(a.tobytes()).hexdigest()


def _cand_key(c: Dict[str, Any]) -> Tuple[str, int]:
    a = np.asarray(c.get("a", []), dtype=np.int32).ravel()
    k = int(c.get("k", 0))
    return (pat_uid(a), k)


# =========================
# C3: conservative dominance / dedup
# =========================
def dominance_prune_modes_C3(
    candidates: List[Dict[str, Any]],
    max_keep: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    C3（保守版）：
    - 只做“(a,k) 完全相同”的去重（稳定 UID）
    - 可选：max_keep 截断（用于控制计算量）
    """
    if not ENABLE_C3:
        return candidates

    seen = set()
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        key = _cand_key(c)
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
        if max_keep is not None and len(kept) >= int(max_keep):
            break
    return kept


# =========================
# Compatibility no-ops (C1/C2/M2/GATE)
# =========================
def dominance_prune_within_candidates(
    candidates: List[Dict[str, Any]],
    d_remain: np.ndarray,
    enable_C1: Optional[bool] = None,
    enable_C2: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    兼容接口：旧代码可能调用 C1/C2。
    目前你决定删除这些策略，所以这里 no-op。
    """
    return candidates


def monotonicity_prune_M2_vs_history(
    candidates: List[Dict[str, Any]],
    history_pool: np.ndarray,
    enable_M2: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    兼容接口：旧代码可能调用 M2。
    目前 no-op。
    """
    return candidates


def gate_candidates(
    candidates: List[Dict[str, Any]],
    enable_gate: Optional[bool] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    兼容接口：旧代码可能调用 gate。
    目前 no-op。
    """
    return candidates


# =========================
# History dedup (stable)
# =========================
def history_dedup(
    candidates: List[Dict[str, Any]],
    history_actions: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    从 candidates 中去掉“历史已经执行过的 (uid,k)”：
    - history_actions: set((uid, k))
      其中 uid 必须是 pat_uid(a)，不能用 Python hash()（不稳定）。
    """
    if not history_actions:
        return candidates

    out: List[Dict[str, Any]] = []
    for c in candidates:
        key = _cand_key(c)
        if key in history_actions:
            continue
        out.append(c)
    return out


# =========================
# Heuristic filter (optional)
# =========================
def heuristic_filter_with_fallback(
    candidates: List[Dict[str, Any]],
    enable: Optional[bool] = None,
    K0: int = 64,
) -> List[Dict[str, Any]]:
    """
    轻量 heuristic：保留 top-K0 个候选，按 trim 升序（trim 越小越好）。
    - enable=None 时：认为“可以做但不强制”，这里默认启用
    - 若 candidates 太少，直接返回全部
    """
    if enable is False:
        return candidates
    if not candidates:
        return candidates

    K0 = max(1, int(K0))
    if len(candidates) <= K0:
        return candidates

    # trim 不一定存在；不存在则给一个大值
    def score(c: Dict[str, Any]) -> Tuple[int, int]:
        trim = c.get("trim", None)
        trim = int(trim) if trim is not None else 10**9
        # 次级排序：k 越大优先（可选）
        k = int(c.get("k", 0))
        return (trim, -k)

    c_sorted = sorted(candidates, key=score)
    kept = c_sorted[:K0]
    if not kept:
        kept = [candidates[0]]
    return kept


# =========================
# Config setter (compat)
# =========================
def set_prune_config(
    enable_c3: bool = True,
    enable_c1: bool = False,
    enable_c2: bool = False,
    enable_m2: bool = False,
    enable_gate: bool = False,
) -> None:
    """
    兼容入口：允许旧脚本通过 set_prune_config 配置剪枝。
    但你当前策略是只留 C3，所以其它开关只是占位。
    """
    global ENABLE_C3, ENABLE_C1, ENABLE_C2, ENABLE_M2, ENABLE_GATE
    ENABLE_C3 = bool(enable_c3)
    ENABLE_C1 = bool(enable_c1)
    ENABLE_C2 = bool(enable_c2)
    ENABLE_M2 = bool(enable_m2)
    ENABLE_GATE = bool(enable_gate)
