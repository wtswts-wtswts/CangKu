# -*- coding: utf-8 -*-
"""
run_dir_instances_rc.py (fixed)

批量跑目录下所有 txt，并调用 run_txt_instances_rc.run_one
修复：原版误用 run_one(args, file_path)，导致缺少大量 positional 参数报错。
"""

import argparse
import os
from run_txt_instances_rc import run_one


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="output")
    p.add_argument("--load-weights", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--solver", type=str, default="gurobi")
    p.add_argument("--timelimit", type=float, default=0.2)
    p.add_argument("--verbose", action="store_true")

    # === 与 run_txt_instances_rc.py 保持一致的参数（补全 run_one 所需） ===
    p.add_argument("--heuristic-topk", action="store_true", help="启用启发式 top-k mask（兼容旧命令）")
    p.add_argument("--topk-K0", type=int, default=64)

    p.add_argument("--pricing-reuse-model", action="store_true", help="启用 pricer cache（兼容旧命令）")
    p.add_argument("--pricer-seed", type=int, default=None)

    # reuse gate
    p.add_argument("--reuse-m-fixed", type=float, default=None, help="若给定，则关闭自适应 m%，使用固定阈值/比例")
    p.add_argument("--reuse-top-pct", type=float, default=0.10)
    p.add_argument("--reuse-m-min", type=float, default=0.05)
    p.add_argument("--reuse-m-max", type=float, default=0.25)
    p.add_argument("--reuse-m-alpha", type=float, default=1.0)

    p.add_argument("--rc-eps", type=float, default=1e-9)

    # ILP predictor small pool
    p.add_argument("--ilp-enable", action="store_true")
    p.add_argument("--ilp-delta", type=int, default=0)
    p.add_argument("--ilp-timelimit", type=float, default=0.5)
    p.add_argument("--ilp-max-cols", type=int, default=250)

    # auto topk/hist controls (与 run_txt_instances_rc 默认一致)
    p.add_argument("--ilp-topk", type=int, default=64)
    p.add_argument("--ilp-topk-mult", type=float, default=3.0)
    p.add_argument("--ilp-topk-min", type=int, default=32)
    p.add_argument("--ilp-topk-max", type=int, default=128)

    p.add_argument("--ilp-hist", type=int, default=64)
    p.add_argument("--ilp-hist-mult", type=float, default=3.0)
    p.add_argument("--ilp-hist-min", type=int, default=32)
    p.add_argument("--ilp-hist-max", type=int, default=128)

    p.add_argument("--ilp-relax-on-infeasible", action="store_true")
    p.add_argument("--ilp-relax-max", type=int, default=2)

    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # run all txt
    for fn in sorted(os.listdir(args.dir)):
        if not fn.lower().endswith(".txt"):
            continue
        file_path = os.path.join(args.dir, fn)

        run_one(
            path=file_path,
            load_weights=args.load_weights,
            device=args.device,
            save_dir=args.save_dir,
            solver=args.solver,
            timelimit=args.timelimit,
            verbose=bool(args.verbose),

            heuristic_topk=bool(args.heuristic_topk),
            topk_K0=int(args.topk_K0),

            pricing_reuse_model=bool(args.pricing_reuse_model),
            pricer_seed=args.pricer_seed,

            reuse_m_fixed=(args.reuse_m_fixed is not None),
            reuse_top_pct=float(args.reuse_top_pct),
            reuse_m_min=float(args.reuse_m_min),
            reuse_m_max=float(args.reuse_m_max),
            reuse_m_alpha=float(args.reuse_m_alpha),
            rc_eps=float(args.rc_eps),

            ilp_enable=bool(args.ilp_enable),
            ilp_delta=int(args.ilp_delta),
            ilp_timelimit=float(args.ilp_timelimit),
            ilp_max_cols=int(args.ilp_max_cols),

            ilp_topk=int(args.ilp_topk),
            ilp_topk_mult=float(args.ilp_topk_mult),
            ilp_topk_min=int(args.ilp_topk_min),
            ilp_topk_max=int(args.ilp_topk_max),

            ilp_hist=int(args.ilp_hist),
            ilp_hist_mult=float(args.ilp_hist_mult),
            ilp_hist_min=int(args.ilp_hist_min),
            ilp_hist_max=int(args.ilp_hist_max),

            ilp_relax_on_infeasible=bool(args.ilp_relax_on_infeasible),
            ilp_relax_max=int(args.ilp_relax_max),
        )


if __name__ == "__main__":
    main()
