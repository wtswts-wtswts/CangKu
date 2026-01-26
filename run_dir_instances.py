# -*- coding: utf-8 -*-
"""
批量运行目录下多个实例（增强版：多列定价 + 列池管理 + 逐实例计时）
做了什么：
- 增加逐实例计时：从解析输入到写完所有输出文件，统计 runtime_sec
- 控制台打印每个实例耗时；summary.json 里也记录 runtime_sec
- 最后打印总耗时与平均耗时，便于做 benchmark
"""

import argparse
import os
import json
import time
from typing import Dict, Any, List
import numpy as np
import torch

from policy import Policy
from rl_rollout import rollout_generate_plan
import pruning as pr
from instance_parser import parse_instance_txt_flexible


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
    counts: Dict[str, int] = {}
    for step in sequence:
        a = np.asarray(step.get('a', []), dtype=int)
        k = int(step.get('k', 0))
        if a.size == 0 or k <= 0:
            continue
        key = pattern_key(a)
        counts[key] = counts.get(key, 0) + k
    return {'counts': counts, 'num_unique_patterns': len(counts),
            'total_bars': int(sum(counts.values()))}


def build_report(w: np.ndarray, d: np.ndarray, L: int, sequence) -> Dict[
    str, Any]:
    w = np.asarray(w, dtype=int)
    d = np.asarray(d, dtype=int)
    m = len(w)

    produced = np.zeros(m, dtype=int)
    total_bars = 0
    total_trim = 0
    total_used_len = 0

    for step in sequence:
        a = np.asarray(step.get('a', []), dtype=int).reshape(-1)
        if a.size == 0:
            continue
        k = int(step.get('k', 0))
        if k <= 0:
            continue
        if a.size != m:
            raise ValueError(
                f"sequence 中模式维度不一致：a.size={a.size} != m={m}")
        produced += k * a
        total_bars += k

        used_len = step.get('used_len', None)
        trim = step.get('trim', None)
        if used_len is None:
            used_len = int(np.inner(w, a))
        if trim is None:
            trim = int(L - int(used_len))

        total_used_len += int(used_len) * k
        total_trim += int(trim) * k

    oversupply = np.maximum(produced - d, 0)
    shortage = np.maximum(d - produced, 0)

    return {
        'L': int(L),
        'm': int(m),
        'total_bars': int(total_bars),
        'total_used_len': int(total_used_len),
        'total_trim': int(total_trim),
        'demand_total_pieces': int(np.sum(d)),
        'produced_total_pieces': int(np.sum(produced)),
        'oversupply_total_pieces': int(np.sum(oversupply)),
        'shortage_total_pieces': int(np.sum(shortage)),
        'is_feasible_cover': bool(np.all(produced >= d)),
    }


def run_one_file(
        file_path: str,
        policy: Policy,
        save_dir: str,
        solver: str,
        timelimit: float,
        heuristic_topk: bool,
        topk_K0: int,
        pricing_reuse_model: bool,
        pricer_seed: int | None,
        pricing_topk: int,
        pricing_noise: float,
        max_cols: int,
        recent_keep: int,
        x_eps: float,
        seed: int | None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    w, d, L, meta = parse_instance_txt_flexible(file_path)

    history_pool, seq, info = rollout_generate_plan(
        w=w, d_init=d, L=L, policy=policy, mode='eval',
        solver=solver, timelimit=timelimit,
        enable_gate=False,
        enable_heuristic_topk=heuristic_topk,
        topk_K0=topk_K0,
        use_pricer_cache=pricing_reuse_model,
        pricer_seed=pricer_seed,
        pricing_topk=pricing_topk,
        pricing_noise=pricing_noise,
        max_cols=max_cols,
        recent_keep=recent_keep,
        x_eps=x_eps,
        seed=seed
    )
    info["parse_meta"] = meta

    counts_info = aggregate_counts(seq)
    report = build_report(w=w, d=d, L=L, sequence=seq)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]

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

    t1 = time.perf_counter()
    runtime_sec = float(t1 - t0)

    return {
        'instance': base,
        'path': file_path,
        'format': meta.get("format"),
        'L': int(L),
        'm': int(len(w)),
        'demand_total_pieces': int(np.sum(d)),
        'bars': int(report['total_bars']),
        'trim': int(report['total_trim']),
        'feasible_cover': bool(report['is_feasible_cover']),
        'shortage_total_pieces': int(report['shortage_total_pieces']),
        'oversupply_total_pieces': int(report['oversupply_total_pieces']),
        'runtime_sec': runtime_sec,
    }


def main():
    ap = argparse.ArgumentParser(
        description="批量运行目录下多个 Cutting Stock / BPP 实例")
    ap.add_argument('--dir', type=str, required=True,
                    help='包含多个 .txt 的目录')
    ap.add_argument('--save-dir', type=str, default='./out_batch',
                    help='输出目录')
    ap.add_argument('--load-weights', type=str, default='',
                    help='权重文件（含 actorNet/criticNet）')
    ap.add_argument('--device', type=str, default='cpu',
                    help='cpu 或 cuda:0 等')
    ap.add_argument('--solver', type=str, default='gurobi',
                    help='定价器（当前第一阶段固定用 gurobi）')
    ap.add_argument('--timelimit', type=float, default=0.2,
                    help='定价器时间上限（秒）')

    ap.add_argument('--pricing-reuse-model', action='store_true',
                    help='开启定价模型复用（更快，默认关闭）')
    ap.add_argument('--pricer-seed', type=int, default=None,
                    help='Gurobi pricer Seed（用于复现）')

    ap.add_argument('--pricing-topk', type=int, default=1,
                    help='每轮定价生成的列数量 K（默认1=原行为）')
    ap.add_argument('--pricing-noise', type=float, default=0.0,
                    help='对偶价扰动幅度（用于多样化列，默认0）')
    ap.add_argument('--max-cols', type=int, default=0,
                    help='列池最大列数（0表示不启用列池管理）')
    ap.add_argument('--recent-keep', type=int, default=200,
                    help='列池管理：保留最近新增列数量')
    ap.add_argument('--x-eps', type=float, default=1e-9,
                    help='列池管理：active set 阈值 x_eps')
    ap.add_argument('--seed', type=int, default=0,
                    help='随机种子（用于定价扰动与其它随机）')

    ap.add_argument('--prune-c1', action='store_true')
    ap.add_argument('--prune-c2', action='store_true')
    ap.add_argument('--prune-c3', action='store_true')
    ap.add_argument('--prune-m2', action='store_true')
    ap.add_argument('--prune-gate', action='store_true')

    ap.add_argument('--heuristic-topk', action='store_true')
    ap.add_argument('--topk-K0', type=int, default=64)

    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise ValueError(f"目录不存在：{args.dir}")

    pr.set_prune_config(
        enable_c1=args.prune_c1,
        enable_c2=args.prune_c2,
        enable_c3=args.prune_c3,
        enable_m2=args.prune_m2,
        enable_gate=args.prune_gate
    )
    print(
        f"[prune] C1={pr.ENABLE_C1}, C2={pr.ENABLE_C2}, C3={pr.ENABLE_C3}, M2={pr.ENABLE_M2}, GATE={pr.ENABLE_GATE}")
    print(
        f"[pricing] reuse_model={bool(args.pricing_reuse_model)}, pricer_seed={args.pricer_seed}")
    print(f"[pricing-topk] K={args.pricing_topk}, noise={args.pricing_noise}")
    print(
        f"[col-mgr] max_cols={args.max_cols}, recent_keep={args.recent_keep}, x_eps={args.x_eps}, seed={args.seed}")

    policy = Policy(device=args.device)
    if args.load_weights and os.path.isfile(args.load_weights):
        ms = torch.load(args.load_weights, map_location=args.device)
        policy.actor.load_state_dict(ms['actorNet'], strict=False)
        policy.critic.load_state_dict(ms['criticNet'], strict=False)
        print(f"[info] 已加载权重：{args.load_weights}")
    else:
        print("[info] 未提供或找不到权重文件，使用随机初始化策略。")

    files = [os.path.join(args.dir, fn) for fn in os.listdir(args.dir) if
             fn.lower().endswith('.txt')]
    files.sort()
    if not files:
        print(f"[提示] 目录下未发现 .txt：{args.dir}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    t_all0 = time.perf_counter()

    for i, fp in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ==> {os.path.basename(fp)}")
        try:
            item = run_one_file(
                file_path=fp,
                policy=policy,
                save_dir=args.save_dir,
                solver=args.solver,
                timelimit=args.timelimit,
                heuristic_topk=args.heuristic_topk,
                topk_K0=args.topk_K0,
                pricing_reuse_model=args.pricing_reuse_model,
                pricer_seed=args.pricer_seed,
                pricing_topk=args.pricing_topk,
                pricing_noise=args.pricing_noise,
                max_cols=args.max_cols,
                recent_keep=args.recent_keep,
                x_eps=args.x_eps,
                seed=args.seed
            )
            summary.append(item)
            print(
                f"  bars={item['bars']}, trim={item['trim']}, feasible={item['feasible_cover']}, "
                f"format={item['format']}, time={item['runtime_sec']:.3f}s")
        except Exception as e:
            err = {
                'instance': os.path.splitext(os.path.basename(fp))[0],
                'path': fp,
                'error': str(e),
            }
            summary.append(err)
            print(f"  [FAILED] {e}")

    t_all1 = time.perf_counter()
    total_sec = float(t_all1 - t_all0)

    out_summary = os.path.join(args.save_dir, "summary.json")
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable({'data': summary}), f, ensure_ascii=False,
                  indent=2)

    # 统计平均耗时（只对成功项）
    rt = [x.get("runtime_sec") for x in summary if
          isinstance(x, dict) and "runtime_sec" in x]
    avg_sec = float(np.mean(rt)) if rt else float("nan")

    print(f"\n[done] summary saved: {out_summary}")
    print(
        f"[time] total={total_sec:.3f}s, avg_per_instance={avg_sec:.3f}s, n={len(files)}")


if __name__ == '__main__':
    main()