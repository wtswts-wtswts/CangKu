# -*- coding: utf-8 -*-
"""
实例解析模块（新增）
做了什么：
- 将 txt 实例解析从脚本中抽离，提供统一入口 parse_instance_txt_flexible
- 自动兼容两种常见格式：
  A) 逐件列表：N, L, 然后每行一个长度
  B) 聚合计数：N, L, 然后每行 length count
- 返回 (w, d, L, meta) 便于写入 info/report 追溯解析细节
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def parse_instance_txt_flexible(path: str) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # 过滤空行；保留原始序号信息可以在 meta 里做，但这里先简单处理
    lines = [ln for ln in lines if ln != ""]

    if len(lines) < 3:
        raise ValueError(f"文件内容不足：{path}")

    N = int(lines[0])
    L = int(lines[1])

    third = lines[2].replace(",", " ").split()
    meta: Dict[str, Any] = {
        "path": path,
        "N_header": int(N),
        "L": int(L),
        "format": None,
        "warnings": [],
    }

    # A) 逐件列表
    if len(third) == 1:
        pieces = []
        for ln in lines[2:]:
            toks = ln.replace(",", " ").split()
            if not toks:
                continue
            try:
                pieces.append(int(toks[0]))
            except Exception:
                continue

        if N != len(pieces):
            meta["warnings"].append(f"N_header={N} 与逐件行数={len(pieces)} 不一致（按实际行数处理）")

        uniq, counts = np.unique(np.array(pieces, dtype=int), return_counts=True)
        meta["format"] = "piece_list"
        meta["N_effective"] = int(len(pieces))
        meta["sum_d"] = int(np.sum(counts))
        meta["m"] = int(len(uniq))
        return uniq.astype(int), counts.astype(int), int(L), meta

    # B) 聚合计数 length count
    if len(third) >= 2:
        lens = []
        cnts = []
        for ln in lines[2:]:
            toks = ln.replace(",", " ").split()
            if len(toks) < 2:
                continue
            try:
                le = int(toks[0])
                ct = int(toks[1])
            except Exception:
                continue
            if ct <= 0:
                continue
            lens.append(le)
            cnts.append(ct)

        w = np.array(lens, dtype=int)
        d = np.array(cnts, dtype=int)
        s = int(np.sum(d))
        if N != s:
            meta["warnings"].append(f"N_header={N} 与 sum(d)={s} 不一致（不强制）")

        meta["format"] = "aggregated_len_count"
        meta["N_effective"] = int(s)
        meta["sum_d"] = int(s)
        meta["m"] = int(len(w))
        return w, d, int(L), meta

    raise ValueError(f"无法识别格式：{path}")