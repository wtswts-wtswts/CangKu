# -*- coding: utf-8 -*-
"""
运行单个/目录 txt 实例生成第一阶段切割方案（rc + RL阈值版）
"""

import argparse
import os
import json
from typing import Dict, Any
import numpy as np
import torch

import pruning1 as pr
from policy1 import Policy
from rl_rollout_rc1 import rollout_generate_plan
from instance_parser1 import parse_instance_txt_flexible


def to_jsonable(obj):
    import numpy as np
    import torch
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def pattern_key(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=int).ravel()
    return ",".join(map(str, a.tolist()))


def aggregate_counts(sequence) -> Dict[str, Any]:
    """bars/patterns/trim 等统计 + 复用计数 counter"""
    total_bars = 0
    uniq = set()
    counter = 0  # 复用的 bars 次数累计（is_reuse=True 的 k 累加）

    for step in sequence:
        k = int(step.get("k", 0))
        total_bars += k
        a = np.asarray(step.get("a", []), dtype=int)
        uniq.add(pattern_key(a))
        if bool(step.get("is_reuse", False)):
            counter += k

    return {
        "total_bars": int(total_bars),
        "num_unique_patterns": int(len(uniq)),
        "counter": int(counter),
    }


def build_report(w: np.ndarray, d: np.ndarray, L: int, sequence) -> Dict[str, Any]:
    """按 demand-based trimloss 计算（与你 fiber 数据集定义一致）"""
    bars = 0
    produced = np.zeros_like(d, dtype=int)

    for st in sequence:
        a = np.asarray(st["a"], dtype=int)
        k = int(st["k"])
        bars += k
        produced += a * k

    demand_sum = float(np.sum(w * d))
    trim_loss_pct = 0.0
    if demand_sum > 0:
        trim_loss_pct = 100.0 * (float(L) * float(bars) - demand_sum) / demand_sum

    return {
        "L": int(L),
        "w": w.tolist(),
        "demand": d.tolist(),
        "bars": int(bars),
        "produced": produced.tolist(),
        "oversupply": (produced - d).tolist(),
        "total_trim": float(L) * float(bars) - float(demand_sum),
        "trim_loss_pct": float(trim_loss_pct),
    }


def run_one(path: str, load_weights: str, device: str, save_dir: str, solver: str, timelimit: float, verbose: bool,
            heuristic_topk: bool, topk_K0: int,
            pricing_reuse_model: bool, pricer_seed: int | None,
            reuse_m_fixed: bool, reuse_top_pct: float, reuse_m_min: float, reuse_m_max: float, reuse_m_alpha: float, rc_eps: float,
            ilp_enable: bool, ilp_delta: int, ilp_timelimit: float, ilp_max_cols: int,
            ilp_topk: int, ilp_topk_mult: float, ilp_topk_min: int, ilp_topk_max: int,
            ilp_hist: int, ilp_hist_mult: float, ilp_hist_min: int, ilp_hist_max: int,
            ilp_relax_on_infeasible: bool, ilp_relax_max: int,
            rl_topk: int, util_min_aux: float, mini_pool_cap: int, mini_pool_max_steps: int,
            mini_ilp_timelimit: float, tail_drop_last: int, tail_util_threshold: float,
            log_trace: bool, trace_json: str):
    print(f"==> 解析文件：{path}")
    # parse_instance_txt_flexible 返回 tuple(w, d, L, meta)
    w, d, L, meta = parse_instance_txt_flexible(path)
    w = np.asarray(w, dtype=int)
    d = np.asarray(d, dtype=int)
    L = int(L)

    print(f"[info] L={L}, 物品种类 m={len(w)}, 总需求件数={int(np.sum(d))} ({meta.get('format')})")
    if meta.get("warnings"):
        for wmsg in meta["warnings"]:
            print(f"[warn] {wmsg}")
    print(f"[pricing] reuse_model={bool(pricing_reuse_model)}, pricer_seed={pricer_seed}")

    policy = Policy(device=device)
    if load_weights and os.path.isfile(load_weights):
        ms = torch.load(load_weights, map_location=device)
        # 兼容旧权重 key
        if isinstance(ms, dict) and ("actorNet" in ms or "criticNet" in ms):
            if "actorNet" in ms:
                policy.actor.load_state_dict(ms["actorNet"], strict=False)
            if "criticNet" in ms:
                policy.critic.load_state_dict(ms["criticNet"], strict=False)
        else:
            # 兼容直接 state_dict 的情况
            try:
                policy.actor.load_state_dict(ms, strict=False)
            except Exception:
                pass
        print(f"[info] 已加载权重：{load_weights}")
    else:
        print("[info] 未提供或找不到权重文件，使用随机初始化策略。")

    history_pool, seq, info = rollout_generate_plan(
        w=w, d_init=d, L=L, policy=policy, mode='eval',
        solver=solver, timelimit=timelimit,
        enable_heuristic_topk=heuristic_topk,
        topk_K0=topk_K0,
        use_pricer_cache=pricing_reuse_model,
        pricer_seed=pricer_seed,
        reuse_top_pct=float(reuse_top_pct),
        reuse_m_adaptive=bool(not reuse_m_fixed),
        reuse_m_min=float(reuse_m_min),
        reuse_m_max=float(reuse_m_max),
        reuse_m_alpha=float(reuse_m_alpha),
        rc_eps=float(rc_eps),
        ilp_enable=bool(ilp_enable),
        ilp_delta=int(ilp_delta),
        ilp_timelimit=float(ilp_timelimit),
        ilp_max_cols=int(ilp_max_cols),
        ilp_topk=int(ilp_topk),
        ilp_topk_mult=float(ilp_topk_mult),
        ilp_topk_min=int(ilp_topk_min),
        ilp_topk_max=int(ilp_topk_max),
        ilp_hist=int(ilp_hist),
        ilp_hist_mult=float(ilp_hist_mult),
        ilp_hist_min=int(ilp_hist_min),
        ilp_hist_max=int(ilp_hist_max),
        ilp_relax_on_infeasible=bool(ilp_relax_on_infeasible),
        ilp_relax_max=int(ilp_relax_max),
        rl_topk=int(rl_topk),
        util_min_aux=float(util_min_aux),
        mini_pool_cap=int(mini_pool_cap),
        mini_pool_max_steps=int(mini_pool_max_steps),
        mini_ilp_timelimit=float(mini_ilp_timelimit),
        tail_drop_last=int(tail_drop_last),
        tail_util_threshold=float(tail_util_threshold),
        log_trace=bool(log_trace),
        trace_json=str(trace_json or ""),
    )
    info["parse_meta"] = meta

    counts_info = aggregate_counts(seq)
    report = build_report(w=w, d=d, L=L, sequence=seq)

    # === 打印 Stage1 指标（你要求的 bars/patterns/trimloss/counter）===
    print(f"[Stage1 | CG+GNN+RL] bars={counts_info['total_bars']}, "
          f"patterns={counts_info['num_unique_patterns']}, "
          f"trimloss(demand-based)={report['trim_loss_pct']:.6f}%, "
          f"counter={counts_info['counter']}")

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_pool = os.path.join(save_dir, f"{base}_pool.npy")
    out_seq_json = os.path.join(save_dir, f"{base}_sequence.json")
    out_counts_json = os.path.join(save_dir, f"{base}_counts.json")
    out_info_json = os.path.join(save_dir, f"{base}_info.json")
    out_report_json = os.path.join(save_dir, f"{base}_report.json")

    np.save(out_pool, history_pool)
    with open(out_seq_json, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(seq), f, ensure_ascii=False, indent=2)
    with open(out_counts_json, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(counts_info), f, ensure_ascii=False, indent=2)
    with open(out_info_json, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(info), f, ensure_ascii=False, indent=2)
    with open(out_report_json, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[ok] 输出：{out_seq_json}")
        print(f"[ok] 输出：{out_counts_json}")
        print(f"[ok] 输出：{out_report_json}")
        print(f"[ok] 输出：{out_info_json}")
        print(f"[ok] 输出：{out_pool}")


def main():
    ap = argparse.ArgumentParser(description="运行 txt 实例生成第一阶段切割方案（输出 sequence + counts + report）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--file', type=str, help='单个 txt 文件路径')
    g.add_argument('--dir', type=str, help='包含多个 txt 文件的目录')

    ap.add_argument('--load-weights', type=str, default='', help='旧工程权重文件（含 actorNet/criticNet）')
    ap.add_argument('--device', type=str, default='cpu', help='cpu 或 cuda:0 等')
    ap.add_argument('--solver', type=str, default='gurobi', help='定价器（当前第一阶段固定用 gurobi）')
    ap.add_argument('--timelimit', type=float, default=0.2, help='定价器时间上限（秒）')
    ap.add_argument('--save-dir', type=str, default='./out', help='结果输出目录')
    ap.add_argument('--verbose', action='store_true', help='打印更详细的提示')

    # ✅ 新增：兼容你命令里的 --prune-c3（当前 pruning 默认已启用 C3）
    ap.add_argument('--prune-c3', action='store_true', help='启用 C3 剪枝（兼容旧命令；当前 pruning 默认已启用）')

    ap.add_argument('--pricing-reuse-model', action='store_true', help='开启定价模型复用（更快，默认关闭）')
    ap.add_argument('--pricer-seed', type=int, default=None, help='Gurobi pricer Seed（用于复现）')

    ap.add_argument('--heuristic-topk', action='store_true', help='开启 rollout 侧的 heuristic TopK（默认关闭）')
    ap.add_argument('--topk-K0', type=int, default=64, help='heuristic TopK 保留数量 K0')

    # reuse gate (logging)
    ap.add_argument('--reuse-m-fixed', action='store_true', help='关闭自适应 m%，改用固定 --reuse-top-pct')
    ap.add_argument('--reuse-top-pct', type=float, default=0.10, help='固定 m%（仅当 --reuse-m-fixed 时生效）')
    ap.add_argument('--reuse-m-min', type=float, default=0.05, help='自适应 m% 下界')
    ap.add_argument('--reuse-m-max', type=float, default=0.25, help='自适应 m% 上界')
    ap.add_argument('--reuse-m-alpha', type=float, default=1.0, help='自适应曲线指数（>1 收紧更快）')
    ap.add_argument('--rc-eps', type=float, default=1e-9, help='rc 判定容差：rc<=0+eps 视为通过')

    # ε-constraint ILP probe
    ap.add_argument('--ilp-enable', action='store_true', help='启用 ε-constraint ILP（bars<=B0+delta, min #patterns）')
    ap.add_argument('--ilp-delta', type=int, default=0, help='ε 约束 slack：bars<=B0+delta')
    ap.add_argument('--ilp-timelimit', type=float, default=0.5, help='ILP 时间上限（秒）')
    ap.add_argument('--ilp-max-cols', type=int, default=250, help='ILP 小列池最大列数（硬上限）')
    ap.add_argument('--ilp-topk', type=int, default=64, help='进入 ILP 的 RL top-k 列数（<=0 则自动）')
    ap.add_argument('--ilp-topk-mult', type=float, default=3.0, help='自动 top-k = mult*m（仅当 ilp-topk<=0）')
    ap.add_argument('--ilp-topk-min', type=int, default=32)
    ap.add_argument('--ilp-topk-max', type=int, default=128)
    ap.add_argument('--ilp-hist', type=int, default=64, help='进入 ILP 的历史列数（<=0 则自动）')
    ap.add_argument('--ilp-hist-mult', type=float, default=3.0, help='自动 hist = mult*m（仅当 ilp-hist<=0）')
    ap.add_argument('--ilp-hist-min', type=int, default=32)
    ap.add_argument('--ilp-hist-max', type=int, default=128)
    ap.add_argument('--ilp-relax-on-infeasible', action='store_true', help='若 infeasible，允许自动放宽 delta（默认关闭）')
    ap.add_argument('--ilp-relax-max', type=int, default=2, help='最多放宽次数（delta+0..delta+relax_max）')

    ap.add_argument('--rl-topk', type=int, default=3, help='每步 RL 候选 top-k')
    ap.add_argument('--util-min-aux', type=float, default=0.8, help='top2/top3 入小列池的最低利用率')
    ap.add_argument('--mini-pool-cap', type=int, default=10, help='小列池容量（完整加入后触发）')
    ap.add_argument('--mini-pool-max-steps', type=int, default=5, help='窗口最大 step 数（达到即触发）')
    ap.add_argument('--mini-ilp-timelimit', type=float, default=0.5, help='小列池 ILP 时间上限（秒）')
    ap.add_argument('--tail-drop-last', type=int, default=1, help='1=启用拆最后一根')
    ap.add_argument('--tail-util-threshold', type=float, default=0.6, help='最后一根利用率阈值')
    ap.add_argument('--log-trace', action='store_true', help='打印窗口 trace 摘要')
    ap.add_argument('--trace-json', type=str, default='', help='窗口 trace 输出 json 路径')

    args = ap.parse_args()

    # pruning config (C3 only). Keep --prune-c3 for backward compatibility.
    if hasattr(args, 'prune_c3') and args.prune_c3:
        pr.ENABLE_C3 = True

    if args.file:
        paths = [args.file]
    else:
        paths = [os.path.join(args.dir, fn) for fn in os.listdir(args.dir) if fn.lower().endswith('.txt')]
        paths.sort()

    for pth in paths:
        run_one(
            pth, args.load_weights, args.device, args.save_dir, args.solver, args.timelimit, args.verbose,
            args.heuristic_topk, args.topk_K0,
            args.pricing_reuse_model, args.pricer_seed,
            args.reuse_m_fixed, args.reuse_top_pct, args.reuse_m_min, args.reuse_m_max, args.reuse_m_alpha, args.rc_eps,
            args.ilp_enable, args.ilp_delta, args.ilp_timelimit, args.ilp_max_cols,
            args.ilp_topk, args.ilp_topk_mult, args.ilp_topk_min, args.ilp_topk_max,
            args.ilp_hist, args.ilp_hist_mult, args.ilp_hist_min, args.ilp_hist_max,
            args.ilp_relax_on_infeasible, args.ilp_relax_max,
            args.rl_topk, args.util_min_aux, args.mini_pool_cap, args.mini_pool_max_steps,
            args.mini_ilp_timelimit, args.tail_drop_last, args.tail_util_threshold,
            args.log_trace, args.trace_json
        )


if __name__ == "__main__":
    main()
