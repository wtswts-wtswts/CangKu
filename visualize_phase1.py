# -*- coding: utf-8 -*-
"""
Phase-1 可视化脚本（单文件）
- 输入：baseline_dir 与 treatment_dir（各自包含 summary.json）
- 输出：一组 png 图到 out_dir
依赖：pandas, numpy, matplotlib

用法示例（Windows）：
python visualize_phase1.py ^
  --baseline-dir "C:\...\out_hard28_baseline" ^
  --treatment-dir "C:\...\out_hard28_topk4" ^
  --out-dir "C:\...\fig_hard28_compare"

可选：只画全局图（默认），目前不读取 per-instance info 曲线。
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_summary(summary_path: str) -> pd.DataFrame:
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"找不到 summary.json：{summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "data" not in obj or not isinstance(obj["data"], list):
        raise ValueError(f"summary.json 格式不对：{summary_path}")
    df = pd.DataFrame(obj["data"])

    # 只保留成功项（有 bars/trim 的）
    need_cols = ["instance", "bars", "trim"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"{summary_path} 缺少字段：{c}")

    ok = df["bars"].notna() & df["trim"].notna()
    df = df.loc[ok].copy()

    # 兼容没有 runtime 的情况
    if "runtime_sec" not in df.columns:
        df["runtime_sec"] = np.nan

    df["bars"] = df["bars"].astype(int)
    df["trim"] = df["trim"].astype(float)
    df["runtime_sec"] = df["runtime_sec"].astype(float)
    df["trim_per_bar"] = df["trim"] / df["bars"].replace(0, np.nan)

    return df


def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def hist_compare(
    a: pd.Series,
    b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    xlabel: str,
    out_path: str,
    bins: int = 20,
):
    plt.figure(figsize=(7.5, 4.5))
    a = a.dropna()
    b = b.dropna()
    plt.hist(a, bins=bins, alpha=0.55, label=label_a, density=False)
    plt.hist(b, bins=bins, alpha=0.55, label=label_b, density=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    savefig(out_path)


def scatter_compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    title: str,
    out_path: str,
):
    plt.figure(figsize=(7.0, 5.5))
    plt.scatter(df_a["bars"], df_a["trim"], s=22, alpha=0.75, label="baseline")
    plt.scatter(df_b["bars"], df_b["trim"], s=22, alpha=0.75, label="treatment")
    plt.title(title)
    plt.xlabel("bars")
    plt.ylabel("trim")
    plt.legend()
    savefig(out_path)


def paired_merge(df_base: pd.DataFrame, df_treat: pd.DataFrame) -> pd.DataFrame:
    # inner join：只比较两边都有的实例
    m = df_base.merge(df_treat, on="instance", how="inner", suffixes=("_base", "_treat"))
    # 计算差分：treat - base（负数表示改善）
    m["delta_bars"] = m["bars_treat"] - m["bars_base"]
    m["delta_trim"] = m["trim_treat"] - m["trim_base"]
    m["delta_trim_per_bar"] = m["trim_per_bar_treat"] - m["trim_per_bar_base"]
    m["delta_time"] = m["runtime_sec_treat"] - m["runtime_sec_base"]
    return m


def plot_delta_sorted(
    merged: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
    out_path: str,
    good_is_negative: bool = True,
    max_instances: int = 200,
):
    # 排序：改善最多的放前面
    m = merged.copy()
    m = m.sort_values(col, ascending=True if good_is_negative else False)
    if len(m) > max_instances:
        m = m.iloc[:max_instances].copy()

    plt.figure(figsize=(10.0, 4.8))
    x = np.arange(len(m))
    y = m[col].values
    plt.bar(x, y)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("instances (sorted)")
    plt.ylabel(ylabel)
    savefig(out_path)


def plot_delta_time_vs_delta_trim(merged: pd.DataFrame, out_path: str):
    """
    权衡图：Δtrim vs Δtime
    - 左下角：trim 更小且更快（最好）
    - 左上角：trim 更小但更慢（质量换时间）
    - 右下角：trim 更差但更快
    - 右上角：两者都更差
    """
    plt.figure(figsize=(7.2, 5.6))
    x = merged["delta_time"].values
    y = merged["delta_trim"].values
    plt.scatter(x, y, s=26, alpha=0.75)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title("Trade-off: Δtime vs Δtrim (treatment - baseline)")
    plt.xlabel("Δtime (sec)  [<0 means faster]")
    plt.ylabel("Δtrim        [<0 means better]")
    savefig(out_path)


def write_quick_stats(merged: pd.DataFrame, out_path: str):
    """
    输出一些关键统计到 txt，便于论文写“整体提升多少”。
    """
    def pct(x):
        return float(100.0 * x)

    n = len(merged)
    improved_trim = int((merged["delta_trim"] < 0).sum())
    worsened_trim = int((merged["delta_trim"] > 0).sum())
    equal_trim = int((merged["delta_trim"] == 0).sum())

    faster = int((merged["delta_time"] < 0).sum())
    slower = int((merged["delta_time"] > 0).sum())

    # 相对改进（trim）
    base_trim_sum = float(merged["trim_base"].sum())
    treat_trim_sum = float(merged["trim_treat"].sum())
    rel_trim_impr = (base_trim_sum - treat_trim_sum) / base_trim_sum if base_trim_sum > 0 else np.nan

    # 相对改进（时间）
    base_time_sum = float(np.nansum(merged["runtime_sec_base"].values))
    treat_time_sum = float(np.nansum(merged["runtime_sec_treat"].values))
    rel_time_impr = (base_time_sum - treat_time_sum) / base_time_sum if base_time_sum > 0 else np.nan

    lines = []
    lines.append("Phase-1 comparison quick stats\n")
    lines.append(f"n_instances_compared: {n}\n")
    lines.append("\n[Trim]\n")
    lines.append(f"improved: {improved_trim} ({pct(improved_trim/n):.1f}%)\n")
    lines.append(f"worsened: {worsened_trim} ({pct(worsened_trim/n):.1f}%)\n")
    lines.append(f"equal:    {equal_trim} ({pct(equal_trim/n):.1f}%)\n")
    lines.append(f"sum_trim_baseline:  {base_trim_sum:.3f}\n")
    lines.append(f"sum_trim_treatment: {treat_trim_sum:.3f}\n")
    lines.append(f"relative_trim_improvement: {pct(rel_trim_impr):.2f}%  (positive means better)\n")

    lines.append("\n[Runtime]\n")
    lines.append(f"faster: {faster} ({pct(faster/n):.1f}%)\n")
    lines.append(f"slower: {slower} ({pct(slower/n):.1f}%)\n")
    lines.append(f"sum_time_baseline:  {base_time_sum:.3f}\n")
    lines.append(f"sum_time_treatment: {treat_time_sum:.3f}\n")
    lines.append(f"relative_time_improvement: {pct(rel_time_impr):.2f}%  (positive means faster)\n")

    # bars（谨慎：你可能希望 bars 不变）
    lines.append("\n[Bars]\n")
    lines.append(f"bars_decreased: {int((merged['delta_bars'] < 0).sum())}\n")
    lines.append(f"bars_increased: {int((merged['delta_bars'] > 0).sum())}\n")
    lines.append(f"bars_equal:     {int((merged['delta_bars'] == 0).sum())}\n")
    lines.append(f"mean_delta_bars: {merged['delta_bars'].mean():.3f}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=str, required=True)
    ap.add_argument("--treatment-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    base_summary = os.path.join(args.baseline_dir, "summary.json")
    treat_summary = os.path.join(args.treatment_dir, "summary.json")

    df_base = load_summary(base_summary)
    df_treat = load_summary(treat_summary)
    merged = paired_merge(df_base, df_treat)

    ensure_out_dir(args.out_dir)

    # 1) bars / trim / trim_per_bar / runtime 分布对比
    hist_compare(
        df_base["bars"], df_treat["bars"],
        "baseline", "treatment",
        "Distribution of bars", "bars",
        os.path.join(args.out_dir, "fig01_bars_hist.png"),
        bins=args.bins
    )

    hist_compare(
        df_base["trim"], df_treat["trim"],
        "baseline", "treatment",
        "Distribution of total trim", "trim",
        os.path.join(args.out_dir, "fig02_trim_hist.png"),
        bins=args.bins
    )

    hist_compare(
        df_base["trim_per_bar"], df_treat["trim_per_bar"],
        "baseline", "treatment",
        "Distribution of trim per bar", "trim/bars",
        os.path.join(args.out_dir, "fig03_trim_per_bar_hist.png"),
        bins=args.bins
    )

    hist_compare(
        df_base["runtime_sec"], df_treat["runtime_sec"],
        "baseline", "treatment",
        "Distribution of runtime (sec)", "runtime_sec",
        os.path.join(args.out_dir, "fig04_runtime_hist.png"),
        bins=args.bins
    )

    # 2) bars vs trim scatter
    scatter_compare(
        df_base, df_treat,
        "Scatter: bars vs trim", os.path.join(args.out_dir, "fig05_scatter_bars_vs_trim.png")
    )

    # 3) per-instance Δtrim（排序）
    plot_delta_sorted(
        merged, "delta_trim",
        "Per-instance Δtrim (treatment - baseline), sorted (negative is better)",
        "Δtrim",
        os.path.join(args.out_dir, "fig06_pairwise_delta_trim.png"),
        good_is_negative=True
    )

    # 4) trade-off 图：Δtime vs Δtrim
    plot_delta_time_vs_delta_trim(
        merged, os.path.join(args.out_dir, "fig07_tradeoff_delta_time_vs_delta_trim.png")
    )

    # 5) quick stats
    write_quick_stats(
        merged, os.path.join(args.out_dir, "quick_stats.txt")
    )

    print(f"[done] figures saved to: {args.out_dir}")
    print(f"[info] compared instances: {len(merged)} (intersection of two summaries)")


if __name__ == "__main__":
    main()