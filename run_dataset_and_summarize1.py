# -*- coding: utf-8 -*-
"""
Batch runner for 3-stage pipeline:
  Stage1: run_txt_instances_rc1.py  -> produces *_pool.npy, *_report.json, *_sequence.json, *_info.json
  Stage2: combined_phase2_then_lb1.py (phase2 + LB to compress patterns)
  Stage3: combined_phase2_then_lb1.py --stage3-enable (epsilon-constraint on patterns, min bars, optional LB)

Outputs:
  - per instance: stage1 outputs in out-dir
  - per instance: stage2/stage3 outputs in out-dir (solution/counts json)
  - summary CSV

NOTE: You told me you don't want stage3. That is fine: just don't pass --stage3-enable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# IO utils
# -----------------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def run_cmd(cmd: List[str], quiet: bool = False) -> Tuple[int, str]:
    """
    Run a subprocess command and capture tail output.
    Returns: (returncode, tail_str)
    """
    import subprocess

    if quiet:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if len(lines) > 200:
                lines = lines[-200:]
        proc.wait()
        tail = "\n".join(lines[-80:])
        return int(proc.returncode), tail
    else:
        proc = subprocess.run(cmd)
        return int(proc.returncode), ""


# -----------------------------
# Metric parsers
# -----------------------------

def try_load_stage1_metrics(out_dir: Path, stem: str) -> Dict[str, Any]:
    """
    Stage1 metrics come from:
      - <stem>_counts.json  : total_bars, num_unique_patterns, counter   (最可靠)
      - <stem>_report.json  : trim_loss_pct, bars (also contains demand/w/L)
      - <stem>_info.json    : optional counter (fallback)
    """
    out = {
        "stage1_bars": None,
        "stage1_patterns": None,
        "stage1_trimloss_pct": None,
        "stage1_counter": None,
    }

    rep = out_dir / f"{stem}_report.json"
    counts = out_dir / f"{stem}_counts.json"
    info = out_dir / f"{stem}_info.json"

    # 1) counts.json (优先：这里才有 num_unique_patterns)
    if counts.exists():
        try:
            c = read_json(counts)
            out["stage1_bars"] = c.get("total_bars", c.get("bars", out["stage1_bars"]))
            out["stage1_patterns"] = c.get("num_unique_patterns", c.get("patterns", out["stage1_patterns"]))
            out["stage1_counter"] = c.get("counter", c.get("reuse_counter", out["stage1_counter"]))
        except Exception:
            pass

    # 2) report.json (主要读 trimloss；bars 也做兜底)
    if rep.exists():
        try:
            j = read_json(rep)
            if out["stage1_bars"] is None:
                out["stage1_bars"] = j.get("bars", j.get("bars_ub", None))
            out["stage1_trimloss_pct"] = j.get("trim_loss_pct", j.get("trimloss_pct", None))
            # 兼容极少数版本把 patterns 写进 report（不是你当前版本）
            if out["stage1_patterns"] is None:
                out["stage1_patterns"] = j.get("patterns", j.get("num_patterns", None))
        except Exception:
            pass

    # 3) info.json (counter 兜底)
    if info.exists() and out["stage1_counter"] is None:
        try:
            j = read_json(info)
            out["stage1_counter"] = j.get("counter", j.get("reuse_counter", None))
        except Exception:
            pass

    return out


def try_load_stage_solution_metrics(sol_path: Path, prefix: str) -> Dict[str, Any]:
    """
    Load stage2/stage3 metrics.

    Preferred source:
      - <out_prefix>_counts.json (stable keys: total_bars, num_unique_patterns,
        trim_loss_pct_demand / trim_loss_pct_realized)

    Fallback:
      - <out_prefix>_solution.json root keys (legacy: bars/objective_patterns/trim_loss_pct/realized_trim_loss_pct)
      - new solution schema: j["final"]{total_bars,num_unique_patterns,trim_loss_pct_demand,...}
    """
    out = {
        f"{prefix}_bars": None,
        f"{prefix}_patterns": None,
        f"{prefix}_trimloss_pct": None,
        f"{prefix}_realized_trimloss_pct": None,
    }

    # 1) counts.json (most robust)
    counts_path = sol_path.with_name(sol_path.name.replace("_solution.json", "_counts.json"))
    if counts_path.exists():
        try:
            c = read_json(counts_path)
            out[f"{prefix}_bars"] = c.get("total_bars", c.get("bars", None))
            out[f"{prefix}_patterns"] = c.get("num_unique_patterns", c.get("patterns", None))
            # demand-based first
            out[f"{prefix}_trimloss_pct"] = c.get(
                "trim_loss_pct_demand",
                c.get("trimloss_demand", c.get("trim_loss_pct", None)),
            )
            out[f"{prefix}_realized_trimloss_pct"] = c.get(
                "trim_loss_pct_realized",
                c.get("realized_trim_loss_pct", None),
            )
            return out
        except Exception:
            pass

    # 2) legacy/new solution.json
    if not sol_path.exists():
        return out

    try:
        j = read_json(sol_path)

        # legacy root fields
        if ("bars" in j) or ("objective_patterns" in j) or ("trim_loss_pct" in j):
            out[f"{prefix}_bars"] = j.get("bars", None)
            out[f"{prefix}_patterns"] = j.get("objective_patterns", j.get("patterns", None))
            out[f"{prefix}_trimloss_pct"] = j.get("trim_loss_pct", None)
            out[f"{prefix}_realized_trimloss_pct"] = j.get("realized_trim_loss_pct", None)
            return out

        # new schema with nested final summary
        final = j.get("final", None)
        if isinstance(final, dict):
            out[f"{prefix}_bars"] = final.get("total_bars", final.get("bars", None))
            out[f"{prefix}_patterns"] = final.get("num_unique_patterns", final.get("patterns", None))
            out[f"{prefix}_trimloss_pct"] = final.get("trim_loss_pct_demand", final.get("trim_loss_pct", None))
            out[f"{prefix}_realized_trimloss_pct"] = final.get("trim_loss_pct_realized", None)
            return out
    except Exception:
        pass

    return out




def try_load_phase2_vs_lb_metrics(sol_path: Path) -> Dict[str, Any]:
    out = {
        "lb_before_bars": None,
        "lb_before_patterns": None,
        "lb_before_trimloss_pct": None,
        "lb_after_bars": None,
        "lb_after_patterns": None,
        "lb_after_trimloss_pct": None,
    }
    if not sol_path.exists():
        return out
    try:
        j = read_json(sol_path)
        p2 = j.get("phase2_summary", {}) if isinstance(j, dict) else {}
        fin = j.get("final", {}) if isinstance(j, dict) else {}
        if isinstance(p2, dict):
            out["lb_before_bars"] = p2.get("total_bars")
            out["lb_before_patterns"] = p2.get("num_unique_patterns")
            out["lb_before_trimloss_pct"] = p2.get("trim_loss_pct_demand")
        if isinstance(fin, dict):
            out["lb_after_bars"] = fin.get("total_bars")
            out["lb_after_patterns"] = fin.get("num_unique_patterns")
            out["lb_after_trimloss_pct"] = fin.get("trim_loss_pct_demand")
    except Exception:
        pass
    return out

def safe_write_csv(csv_path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> Path:
    """
    Write CSV safely:
      - write to temp then replace
      - if PermissionError (e.g., Excel locking), write to timestamped file
    """
    safe_mkdir(csv_path.parent)
    tmp = csv_path.with_suffix(".tmp.csv")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        tmp.replace(csv_path)
        return csv_path
    except PermissionError:
        alt = csv_path.parent / f"{csv_path.stem}_{now_stamp()}.csv"
        with alt.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return alt


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", required=True, help="Folder containing *.txt instances")
    p.add_argument("--out-dir", required=True, help="Output folder (all per-instance outputs go here)")
    p.add_argument("--csv-out", required=True, help="Path to summary csv")
    p.add_argument("--weights", required=True, help="RL model weights path")
    p.add_argument("--device", default="cpu")

    # Stage1 args
    p.add_argument("--stage1-timelimit", type=float, default=0.2)
    p.add_argument("--solver", default="gurobi", choices=["gurobi", "cbc"])
    p.add_argument("--prune-c3", action="store_true")

    # ILP reuse options (stage1 internal)
    p.add_argument("--ilp-enable", action="store_true")
    p.add_argument("--ilp-delta", type=int, default=0)
    p.add_argument("--ilp-timelimit", type=float, default=0.5)
    p.add_argument("--ilp-topk", type=int, default=64)
    p.add_argument("--ilp-hist", type=int, default=64)
    p.add_argument("--ilp-max-cols", type=int, default=250)

    # reuse gate options (stage1 internal)
    p.add_argument("--reuse-m-min", type=float, default=0.05)
    p.add_argument("--reuse-m-max", type=float, default=0.25)
    p.add_argument("--reuse-m-alpha", type=float, default=1.0)
    p.add_argument("--reuse-m-tau", type=float, default=None)
    p.add_argument("--rc-eps", type=float, default=1e-9)

    p.add_argument("--rl-topk", type=int, default=3)
    p.add_argument("--util-min-aux", type=float, default=0.8)
    p.add_argument("--mini-pool-cap", type=int, default=10)
    p.add_argument("--mini-pool-max-steps", type=int, default=5)
    p.add_argument("--mini-ilp-timelimit", type=float, default=0.5)
    p.add_argument("--tail-drop-last", type=int, default=1)
    p.add_argument("--tail-util-threshold", type=float, default=0.6)

    # Stage2 args
    p.add_argument("--bars-delta", type=int, default=0)
    p.add_argument("--phase2-timelimit", type=float, default=0.5)
    p.add_argument("--k-list", type=str, default="3,5,8,12")
    p.add_argument("--time-per-k", type=float, default=1.0)

    # Stage3 (optional)
    p.add_argument("--stage3-enable", action="store_true")
    p.add_argument("--stage3-patterns-delta", type=int, default=0)
    p.add_argument("--stage3-bars-delta", type=int, default=0)
    p.add_argument("--stage3-timelimit", type=float, default=0.5)
    p.add_argument("--stage3-k-list", type=str, default="3,5,8,12")
    p.add_argument("--stage3-time-per-k", type=float, default=1.0)

    # logging
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress", action="store_true")

    return p


def main():
    args = build_parser().parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".txt"])
    total = len(files)
    if total == 0:
        print(f"未找到 .txt 实例：{data_dir}")
        return

    py = sys.executable

    rows: List[Dict[str, Any]] = []

    for idx, fp in enumerate(files, 1):
        stem = fp.stem
        t0 = time.time()

        if args.progress:
            print(f"[{stem}] start")

        # ---------------- Stage1 ----------------
        cmd1 = [
            py, "run_txt_instances_rc1.py",
            "--file", str(fp),
            "--save-dir", str(out_dir),
            "--load-weights", str(args.weights),
            "--device", str(args.device),
            "--solver", str(args.solver),
            "--timelimit", str(args.stage1_timelimit),
        ]
        if args.prune_c3:
            cmd1.append("--prune-c3")

        if args.ilp_enable:
            cmd1 += [
                "--ilp-enable",
                "--ilp-delta", str(args.ilp_delta),
                "--ilp-timelimit", str(args.ilp_timelimit),
                "--ilp-topk", str(args.ilp_topk),
                "--ilp-hist", str(args.ilp_hist),
                "--ilp-max-cols", str(args.ilp_max_cols),
            ]

        cmd1 += [
            "--reuse-m-min", str(args.reuse_m_min),
            "--reuse-m-max", str(args.reuse_m_max),
            "--reuse-m-alpha", str(args.reuse_m_alpha),
            "--rc-eps", str(args.rc_eps),
            "--rl-topk", str(args.rl_topk),
            "--util-min-aux", str(args.util_min_aux),
            "--mini-pool-cap", str(args.mini_pool_cap),
            "--mini-pool-max-steps", str(args.mini_pool_max_steps),
            "--mini-ilp-timelimit", str(args.mini_ilp_timelimit),
            "--tail-drop-last", str(args.tail_drop_last),
            "--tail-util-threshold", str(args.tail_util_threshold),
        ]
        if args.reuse_m_tau is not None:
            cmd1 += ["--reuse-m-tau", str(args.reuse_m_tau)]

        if args.verbose:
            cmd1.append("--verbose")

        rc1, tail1 = run_cmd(cmd1, quiet=args.quiet)
        if rc1 != 0:
            rows.append({
                "instance": stem,
                "status": "stage1_fail",
                "runtime_sec": f"{time.time() - t0:.3f}",
            })
            print(f"[{idx}/{total}] {stem}: stage1 FAIL (rc={rc1})")
            if tail1:
                print(tail1)
            continue
        if args.progress:
            print(f"[{stem}] stage1 OK")

        # ---------------- Stage2 (+ optional Stage3) ----------------
        pool_path = out_dir / f"{stem}_pool.npy"
        rep_path = out_dir / f"{stem}_report.json"
        seq_path = out_dir / f"{stem}_sequence.json"

        out_prefix = out_dir / f"{stem}_phase2lb"
        cmd2 = [
            py, "combined_phase2_then_lb1.py",
            "--mode", "combined",
            "--pool", str(pool_path),
            "--report", str(rep_path),
            "--sequence", str(seq_path),
            "--bars-delta", str(args.bars_delta),
            "--phase2-timelimit", str(args.phase2_timelimit),
            "--k-list", str(args.k_list),
            "--time-per-k", str(args.time_per_k),
            "--out-prefix", str(out_prefix),
        ]
        if args.verbose:
            cmd2.append("--verbose")

        if args.stage3_enable:
            cmd2 += [
                "--stage3-enable",
                "--stage3-patterns-delta", str(args.stage3_patterns_delta),
                "--stage3-bars-delta", str(args.stage3_bars_delta),
                "--stage3-timelimit", str(args.stage3_timelimit),
                "--stage3-k-list", str(args.stage3_k_list),
                "--stage3-time-per-k", str(args.stage3_time_per_k),
            ]

        rc2, tail2 = run_cmd(cmd2, quiet=args.quiet)
        if rc2 != 0:
            rows.append({
                "instance": stem,
                "status": "stage2_or_3_fail",
                "runtime_sec": f"{time.time() - t0:.3f}",
            })
            print(f"[{idx}/{total}] {stem}: stage2/stage3 FAIL (rc={rc2})")
            if tail2:
                print(tail2)
            continue
        if args.progress:
            print(f"[{stem}] stage2 OK")

        # ---------------- Parse metrics ----------------
        stage1 = try_load_stage1_metrics(out_dir, stem)

        s2_sol = out_dir / f"{stem}_phase2lb_solution.json"
        s3_sol = out_dir / f"{stem}_phase2lb_stage3_solution.json"

        stage2 = try_load_stage_solution_metrics(s2_sol, "stage2")
        stage3 = try_load_stage_solution_metrics(s3_sol, "stage3") if args.stage3_enable else {
            "stage3_bars": None, "stage3_patterns": None, "stage3_trimloss_pct": None, "stage3_realized_trimloss_pct": None
        }

        dt = time.time() - t0
        lb_cmp = try_load_phase2_vs_lb_metrics(s2_sol)
        row = {
            "instance": stem,
            "status": "ok",
            "runtime_sec": f"{dt:.3f}",

            "stage1_bars": stage1["stage1_bars"],
            "stage1_patterns": stage1["stage1_patterns"],
            "stage1_trimloss_pct": stage1["stage1_trimloss_pct"],
            "stage1_counter": stage1["stage1_counter"],

            "lb_before_bars": lb_cmp["lb_before_bars"],
            "lb_before_patterns": lb_cmp["lb_before_patterns"],
            "lb_before_trimloss_pct": lb_cmp["lb_before_trimloss_pct"],
            "lb_after_bars": lb_cmp["lb_after_bars"],
            "lb_after_patterns": lb_cmp["lb_after_patterns"],
            "lb_after_trimloss_pct": lb_cmp["lb_after_trimloss_pct"],

            "stage2_bars": stage2["stage2_bars"],
            "stage2_patterns": stage2["stage2_patterns"],
            "stage2_trimloss_pct": stage2["stage2_trimloss_pct"],

            "stage3_bars": stage3["stage3_bars"],
            "stage3_patterns": stage3["stage3_patterns"],
            "stage3_trimloss_pct": stage3["stage3_trimloss_pct"],
        }
        rows.append(row)

        print(f"[{idx}/{total}] {stem}: ok")

    # ---------------- Write CSV ----------------
    fieldnames = [
        "instance", "status", "runtime_sec",
        "stage1_bars", "stage1_patterns", "stage1_trimloss_pct", "stage1_counter",
        "lb_before_bars", "lb_before_patterns", "lb_before_trimloss_pct",
        "lb_after_bars", "lb_after_patterns", "lb_after_trimloss_pct",
        "stage2_bars", "stage2_patterns", "stage2_trimloss_pct",
        "stage3_bars", "stage3_patterns", "stage3_trimloss_pct",
    ]

    out_csv = safe_write_csv(Path(args.csv_out), rows, fieldnames)
    print(f"\n写出 CSV 完成：{out_csv}")


if __name__ == "__main__":
    main()
