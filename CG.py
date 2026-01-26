# -*- coding: utf-8 -*-
"""
CG（Column Generation，列生成）核心模块（增强版：多列定价 + 列池管理）
做了什么：
1) 支持 multi-column pricing：每轮 LP 后可生成并加入 K 个新列（通过扰动对偶价产生多样化）
2) 支持 column management：限制列池最大列数 max_cols，超限时保留 active set + 最近新增列，删除弱列
3) 保持向后兼容：默认 pricing_topk=1 且 max_cols=0 时行为与之前一致
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import gurobipy as grb

from pattern_cache import PatternPool
from pricing_reuse import BoundedKnapsackPricer


def knapsack_pricing_gurobi_bounded_int(
    pi: np.ndarray,
    w: np.ndarray,
    L: int,
    d_remain: Optional[np.ndarray] = None,
    timelimit: float = 0.2
) -> Tuple[np.ndarray, float]:
    pi = np.asarray(pi, dtype=float)
    w = np.asarray(w, dtype=int)
    n = len(w)

    m = grb.Model("pricing_bounded_int_knapsack")
    m.Params.OutputFlag = 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)

    if d_remain is not None:
        ub = np.asarray(d_remain, dtype=int).clip(min=0)
        if int(ub.sum()) == 0:
            return np.zeros(n, dtype=int), 0.0
    else:
        ub = None

    a = m.addMVar(shape=n, vtype=grb.GRB.INTEGER, lb=0, name="a")
    if ub is not None:
        m.addConstr(a <= ub)

    m.addConstr((w @ a) <= int(L))
    m.setObjective(pi @ a, grb.GRB.MAXIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return np.zeros(n, dtype=int), 0.0

    sol = np.array(a.X, dtype=float).round().astype(int)
    obj = float(pi @ sol)
    return sol.astype(int), obj


class CGRunner:
    def __init__(
        self,
        w: np.ndarray,
        L: int,
        use_pricer_cache: bool = False,
        pricer_seed: Optional[int] = None
    ):
        self.w = np.asarray(w, dtype=int)
        self.L = int(L)
        self.use_pricer_cache = bool(use_pricer_cache)
        self.pricer_seed = pricer_seed

        self._pricer: Optional[BoundedKnapsackPricer] = None
        if self.use_pricer_cache:
            self._pricer = BoundedKnapsackPricer(w=self.w, L=self.L, timelimit=0.2, seed=pricer_seed, output_flag=0)

    @staticmethod
    def _trim_of(a: np.ndarray, w: np.ndarray, L: int) -> int:
        return int(L - int(np.inner(w, a)))

    def build_initial_pool(self, d_remain: Optional[np.ndarray] = None) -> np.ndarray:
        m = len(self.w)
        cols = []
        for i in range(m):
            wi = int(self.w[i])
            if wi <= 0 or wi > self.L:
                continue
            a = np.zeros(m, dtype=int)
            max_cnt = int(self.L // wi)
            if d_remain is not None:
                max_cnt = min(max_cnt, int(max(0, d_remain[i])))
            if max_cnt > 0:
                a[i] = max_cnt
                cols.append(a)

        if len(cols) == 0:
            return np.zeros((m, 0), dtype=int)

        pool = PatternPool(m=m, P=np.zeros((m, 0), dtype=int))
        for a in cols:
            pool.try_add(a)
        return pool.to_numpy().astype(int)

    def solve_lp_master_over(self, P: np.ndarray, d_remain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.asarray(P, dtype=int)
        d_remain = np.asarray(d_remain, dtype=int)

        if P.shape[1] == 0:
            return np.zeros(P.shape[0], dtype=float), np.zeros(0, dtype=float)

        m = grb.Model("LP_Master")
        m.Params.OutputFlag = 0
        mP = P.astype(float)
        m_items = d_remain.shape[0]
        n_cols = P.shape[1]

        x = m.addMVar(shape=n_cols, lb=0.0, vtype=grb.GRB.CONTINUOUS, name="x")
        m.addConstr(mP @ x >= d_remain, name="cover")
        m.setObjective(x.sum(), grb.GRB.MINIMIZE)
        m.optimize()

        cons = m.getConstrs()
        pi = np.array(m.getAttr('Pi', cons)[:m_items], dtype=float)
        frac = np.array(x.X, dtype=float)
        return pi, frac

    def _solve_pricing_one(
        self,
        pi_mod: np.ndarray,
        d_remain: np.ndarray,
        timelimit: float
    ) -> np.ndarray:
        """
        解一次背包定价，返回列 a（整数向量）
        """
        if self.use_pricer_cache and self._pricer is not None:
            a_new, _ = self._pricer.solve(pi=pi_mod, d_remain=d_remain, timelimit=timelimit)
            return a_new.astype(int)

        a_new, _ = knapsack_pricing_gurobi_bounded_int(
            pi=pi_mod, w=self.w, L=self.L, d_remain=d_remain, timelimit=timelimit
        )
        return a_new.astype(int)

    def price_topk(
        self,
        pi: np.ndarray,
        d_remain: np.ndarray,
        gate_alpha: Optional[np.ndarray] = None,
        timelimit: float = 0.2,
        pricing_topk: int = 1,
        pricing_noise: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Dict[str, Any]]:
        """
        multi-column pricing：一次性生成最多 K 个候选列（带多样化扰动）
        返回列表：[{a, rc, trim, pi_noise_scale}, ...]
        - 第 1 个列使用原 pi_mod（不扰动），保证不变差
        - 后续列使用 pi_mod 的随机扰动版本以提高多样性
        """
        pi = np.asarray(pi, dtype=float)
        d_remain = np.asarray(d_remain, dtype=int)

        if gate_alpha is not None:
            alpha = np.asarray(gate_alpha, dtype=float)
            assert alpha.shape == pi.shape
            pi_base = pi * alpha
        else:
            pi_base = pi

        K = int(max(1, pricing_topk))
        noise = float(max(0.0, pricing_noise))
        if rng is None:
            rng = np.random.default_rng(0)

        out: List[Dict[str, Any]] = []
        for t in range(K):
            if t == 0 or noise <= 1e-12:
                pi_mod = pi_base
                scale = 0.0
            else:
                # 对偶价扰动：产生多样化列
                # 为避免目标翻转/极端值，用截断噪声
                z = rng.normal(loc=0.0, scale=1.0, size=pi_base.shape)
                z = np.clip(z, -2.0, 2.0)
                pi_mod = pi_base * (1.0 + noise * z)
                scale = noise

            a_new = self._solve_pricing_one(pi_mod=pi_mod, d_remain=d_remain, timelimit=timelimit)
            if int(a_new.sum()) <= 0:
                continue

            rc = float(1.0 - float(np.inner(pi, a_new)))  # reduced cost 用原 pi 计算
            trim = self._trim_of(a_new, self.w, self.L)

            out.append({'a': a_new.astype(int), 'rc': rc, 'trim': trim, 'pi_noise_scale': scale})

        return out

    @staticmethod
    def _apply_column_management(
        P: np.ndarray,
        frac: np.ndarray,
        max_cols: int,
        recent_keep: int = 200,
        x_eps: float = 1e-9,
    ) -> np.ndarray:
        """
        列池管理：限制列数，保留 active set + 最近新增列，删除弱列
        """
        P = np.asarray(P, dtype=int)
        n = int(P.shape[1])
        max_cols = int(max_cols)

        if max_cols <= 0 or n <= max_cols:
            return P

        frac = np.asarray(frac, dtype=float).reshape(-1)
        if frac.size != n:
            # 保守处理：若维度不一致，不做管理
            return P

        keep = set()

        # 1) 保留 LP 当前使用的列（active set）
        active_idx = np.where(frac > float(x_eps))[0].tolist()
        keep.update(active_idx)

        # 2) 保留最近新增的 recent_keep 列（索引靠后）
        rk = int(max(0, recent_keep))
        if rk > 0:
            keep.update(list(range(max(0, n - rk), n)))

        # 如果 keep 已经超过 max_cols，优先保留 active 中 x 最大的
        keep_list = list(keep)
        if len(keep_list) > max_cols:
            # 按 frac 降序选 max_cols 个
            keep_list.sort(key=lambda j: frac[j], reverse=True)
            keep_list = keep_list[:max_cols]
        else:
            # 否则从剩余列里按 frac 最大补齐（保持多样性和质量）
            if len(keep_list) < max_cols:
                remain = [j for j in range(n) if j not in keep]
                remain.sort(key=lambda j: frac[j], reverse=True)
                need = max_cols - len(keep_list)
                keep_list.extend(remain[:need])

        keep_list = sorted(set(keep_list))
        P_new = P[:, keep_list]
        return P_new

    def run_until_no_neg_rc(
        self,
        P: np.ndarray,
        d_remain: np.ndarray,
        gate_alpha: Optional[np.ndarray] = None,
        solver: str = "gurobi",
        timelimit: float = 0.2,
        max_inner_iters: int = 200,
        pricing_topk: int = 1,
        pricing_noise: float = 0.0,
        max_cols: int = 0,
        recent_keep: int = 200,
        x_eps: float = 1e-9,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        内循环：LP + 定价，直到无负 reduced cost 列
        增强：一次可加入 K 列；可做列池管理
        """
        if solver != "gurobi":
            raise ValueError(f"未知定价器类型：{solver}（当前第一阶段采用 gurobi）")

        P = np.asarray(P, dtype=int)
        d_remain = np.asarray(d_remain, dtype=int)
        added: List[Dict[str, Any]] = []

        if P.shape[1] == 0:
            P = self.build_initial_pool(d_remain=d_remain)

        pool = PatternPool.from_P(P)

        for _ in range(int(max_inner_iters)):
            P_now = pool.to_numpy()

            pi, frac = self.solve_lp_master_over(P_now, d_remain)

            # 列池管理（在定价前做一次也行，这里选择 LP 后做管理）
            if max_cols and int(P_now.shape[1]) > int(max_cols):
                P_managed = self._apply_column_management(
                    P=P_now, frac=frac, max_cols=max_cols, recent_keep=recent_keep, x_eps=x_eps
                )
                pool = PatternPool.from_P(P_managed)
                P_now = pool.to_numpy()
                # 重新解 LP（因为列被删了）
                pi, frac = self.solve_lp_master_over(P_now, d_remain)

            # multi-column pricing
            priced = self.price_topk(
                pi=pi,
                d_remain=d_remain,
                gate_alpha=gate_alpha,
                timelimit=timelimit,
                pricing_topk=pricing_topk,
                pricing_noise=pricing_noise,
                rng=rng
            )

            # 从 priced 中挑出负 rc 的列加入（去重后）
            any_added = False
            for rec in priced:
                a_new = rec['a'].astype(int)
                rc = float(rec['rc'])
                trim = int(rec['trim'])
                if rc < -1e-8 and int(a_new.sum()) > 0:
                    ok = pool.try_add(a_new)
                    added.append({'a': a_new, 'rc': rc, 'trim': trim, 'dedup_skipped': (not ok)})
                    if ok:
                        any_added = True

            if any_added:
                continue

            # 没有新增列 => 退出
            return pool.to_numpy(), pi, frac, added

        # 达到迭代上限：返回当前结果
        pi, frac = self.solve_lp_master_over(pool.to_numpy(), d_remain)
        return pool.to_numpy(), pi, frac, added