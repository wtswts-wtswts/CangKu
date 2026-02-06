# -*- coding: utf-8 -*-
"""
run_end2end.py (latest)

End-to-end convenience runner:
- Stage1: run_txt_instances_rc over a directory (or a single file).
- Stage2 (optional): invoke combined_phase2_then_lb.py externally (kept as user workflow).

This script mainly exists to keep your previous entrypoint stable.
"""
import argparse
import os
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True)
    p.add_argument("--out", type=str, default="output")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--timelimit", type=float, default=0.2)
    p.add_argument("--solver", type=str, default="gurobi")

    # stage1 flags
    p.add_argument("--prune-c3", action="store_true")
    p.add_argument("--ilp-enable", action="store_true")
    p.add_argument("--ilp-delta", type=int, default=0)
    p.add_argument("--ilp-timelimit", type=float, default=0.5)
    p.add_argument("--ilp-topk", type=int, default=64)
    p.add_argument("--ilp-hist", type=int, default=64)
    p.add_argument("--ilp-max-cols", type=int, default=250)
    p.add_argument("--reuse-m-min", type=float, default=0.05)
    p.add_argument("--reuse-m-max", type=float, default=0.25)
    p.add_argument("--reuse-m-alpha", type=float, default=1.0)
    p.add_argument("--reuse-m-fixed", type=float, default=None)
    p.add_argument("--rc-eps", type=float, default=1e-9)

    args = p.parse_args()

    cmd = [
        sys.executable, "run_txt_instances_rc.py",
        "--dir", args.dir,
        "--save-dir", args.out,
        "--load-weights", args.weights,
        "--device", args.device,
        "--solver", args.solver,
        "--timelimit", str(args.timelimit),
        "--rc-eps", str(args.rc_eps),
        "--ilp-delta", str(args.ilp_delta),
        "--ilp-timelimit", str(args.ilp_timelimit),
        "--ilp-topk", str(args.ilp_topk),
        "--ilp-hist", str(args.ilp_hist),
        "--ilp-max-cols", str(args.ilp_max_cols),
        "--reuse-m-min", str(args.reuse_m_min),
        "--reuse-m-max", str(args.reuse_m_max),
        "--reuse-m-alpha", str(args.reuse_m_alpha),
    ]
    if args.reuse_m_fixed is not None:
        cmd += ["--reuse-m-fixed", str(args.reuse_m_fixed)]
    if args.prune_c3:
        cmd += ["--prune-c3"]
    if args.ilp_enable:
        cmd += ["--ilp-enable"]

    os.makedirs(args.out, exist_ok=True)
    print("[run_end2end] Stage1 cmd:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
