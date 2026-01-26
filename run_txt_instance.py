# -*- coding: utf-8 -*-
"""
运行单个/目录 txt 实例生成第一阶段切割方案（修改版）
做了什么：
1) 使用 instance_parser.parse_instance_txt_flexible 统一解析两种格式（逐件/len-count）
2) 增加定价模型复用开关：--pricing-reuse-model（默认关闭）
3) 其余输出文件格式不变，便于与你旧工程继续对齐对比
"""

import argparse
import os
import json
from typing import Tuple, Dict, Any
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
    return {'counts': counts, 'num_unique_patterns': len(counts), 'total_bars': int(sum(counts.values()))}


def build_report(w: np.ndarray, d: np.ndarray, L: int, sequence) -> Dict[str, Any]:
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
            raise ValueError(f"sequence 中模式维度不一致：a.size={a.size} != m={m}")
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

    report = {
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
        'w': w.tolist(),
        'demand': d.tolist(),
        'produced': produced.tolist(),
        'oversupply': oversupply.tolist(),
        'shortage': shortage.tolist(),
    }
    return report


def run_one(path: str, load_weights: str, device: str, save_dir: str, solver: str, timelimit: float, verbose: bool,
            prune_c1: bool, prune_c2: bool, prune_c3: bool, prune_m2: bool, prune_gate: bool,
            heuristic_topk: bool, topk_K0: int,
            pricing_reuse_model: bool, pricer_seed: int | None):
    print(f"==> 解析文件：{path}")
    w, d, L, meta = parse_instance_txt_flexible(path)
    print(f"[info] L={L}, 物品种类 m={len(w)}, 总需求件数={int(np.sum(d))} ({meta.get('format')})")
    if meta.get("warnings"):
        for wmsg in meta["warnings"]:
            print(f"[warn] {wmsg}")

    pr.set_prune_config(
        enable_c1=prune_c1,
        enable_c2=prune_c2,
        enable_c3=prune_c3,
        enable_m2=prune_m2,
        enable_gate=prune_gate
    )
    print(f"[prune] C1={pr.ENABLE_C1}, C2={pr.ENABLE_C2}, C3={pr.ENABLE_C3}, M2={pr.ENABLE_M2}, GATE={pr.ENABLE_GATE}")
    print(f"[pricing] reuse_model={bool(pricing_reuse_model)}, pricer_seed={pricer_seed}")

    policy = Policy(device=device)
    if load_weights and os.path.isfile(load_weights):
        ms = torch.load(load_weights, map_location=device)
        policy.actor.load_state_dict(ms['actorNet'], strict=False)
        policy.critic.load_state_dict(ms['criticNet'], strict=False)
        print(f"[info] 已加载权重：{load_weights}")
    else:
        print("[info] 未提供或找不到权重文件，使用随机初始化策略。")

    history_pool, seq, info = rollout_generate_plan(
        w=w, d_init=d, L=L, policy=policy, mode='eval',
        solver=solver, timelimit=timelimit,
        enable_gate=False,
        enable_heuristic_topk=heuristic_topk,
        topk_K0=topk_K0,
        use_pricer_cache=pricing_reuse_model,
        pricer_seed=pricer_seed
    )
    info["parse_meta"] = meta

    counts_info = aggregate_counts(seq)
    report = build_report(w=w, d=d, L=L, sequence=seq)

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

    print(f"[done] 已保存：\n"
          f"  模式池: {out_pool}\n"
          f"  动作序列: {out_seq_json}\n"
          f"  聚合用量: {out_counts_json}\n"
          f"  运行信息: {out_info_json}\n"
          f"  验收报告: {out_report_json}")

    print(f"[pool] 历史模式池形状: {history_pool.shape}")
    print(f"[seq] 步数: {len(seq)}")
    print(f"[counts] unique patterns: {counts_info['num_unique_patterns']}, total bars: {counts_info['total_bars']}")
    print(f"[report] feasible_cover={report['is_feasible_cover']}, "
          f"shortage_total_pieces={report['shortage_total_pieces']}, "
          f"oversupply_total_pieces={report['oversupply_total_pieces']}, "
          f"total_trim={report['total_trim']}")


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

    ap.add_argument('--pricing-reuse-model', action='store_true', help='开启定价模型复用（更快，默认关闭）')
    ap.add_argument('--pricer-seed', type=int, default=None, help='Gurobi pricer Seed（用于复现）')

    ap.add_argument('--prune-c1', action='store_true', help='开启 C1（跨根数支配）')
    ap.add_argument('--prune-c2', action='store_true', help='开启 C2（同根数支配）')
    ap.add_argument('--prune-c3', action='store_true', help='开启 C3（模式级支配）')
    ap.add_argument('--prune-m2', action='store_true', help='开启 M2（历史单调性剪枝）')
    ap.add_argument('--prune-gate', action='store_true', help='开启 GATE（启发式 Top-K + 回退）')

    ap.add_argument('--heuristic-topk', action='store_true', help='开启 rollout 侧的 heuristic TopK（配合 pruning gate）')
    ap.add_argument('--topk-K0', type=int, default=64, help='heuristic TopK 保留数量 K0')

    args = ap.parse_args()

    def _run(fp: str):
        run_one(
            fp, args.load_weights, args.device, args.save_dir, args.solver, args.timelimit, args.verbose,
            prune_c1=args.prune_c1, prune_c2=args.prune_c2, prune_c3=args.prune_c3, prune_m2=args.prune_m2, prune_gate=args.prune_gate,
            heuristic_topk=args.heuristic_topk, topk_K0=args.topk_K0,
            pricing_reuse_model=args.pricing_reuse_model, pricer_seed=args.pricer_seed
        )

    if args.file:
        _run(args.file)
    else:
        if not os.path.isdir(args.dir):
            print(f"[错误] 目录不存在：{args.dir}")
            return
        files = [os.path.join(args.dir, fn) for fn in os.listdir(args.dir) if fn.lower().endswith('.txt')]
        if not files:
            print(f"[提示] 目录下未发现 .txt：{args.dir}")
            return
        files.sort()
        for fp in files:
            try:
                _run(fp)
            except Exception as e:
                print(f"[跳过] {fp} 失败：{e}")


if __name__ == '__main__':
    main()