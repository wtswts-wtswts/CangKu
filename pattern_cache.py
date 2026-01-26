# -*- coding: utf-8 -*-
"""
模式池/列去重模块（新增）
做了什么：
- 提供 PatternPool：用 bytes 哈希对 pattern 向量去重（避免重复列进入池子和 LP）
- 既可用于 CG 的列池 P 去重，也可用于 history_column_pool 的记录去重
- 去重不改变数学结果（重复列是冗余），通常提升速度与稳定性
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


def _pattern_bytes(a: np.ndarray) -> bytes:
    a = np.asarray(a, dtype=np.int32).ravel()
    return a.tobytes()


@dataclass
class PatternPool:
    """
    一个简单的列池容器，内部存列矩阵 P: shape [m, n_cols]（int）
    并维护 seen 集合用于去重。
    """
    m: int
    P: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    seen: set = field(default_factory=set)

    @classmethod
    def from_P(cls, P: np.ndarray) -> "PatternPool":
        P = np.asarray(P, dtype=int)
        obj = cls(m=P.shape[0], P=P.copy(), seen=set())
        for j in range(P.shape[1]):
            obj.seen.add(_pattern_bytes(P[:, j]))
        return obj

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.P, dtype=int)

    def try_add(self, a: np.ndarray) -> bool:
        """
        尝试添加一列 a（shape [m]）
        返回 True 表示成功新增；False 表示重复被跳过
        """
        a = np.asarray(a, dtype=int).reshape(-1)
        if a.size != self.m:
            raise ValueError(f"pattern 维度不一致：a.size={a.size} != m={self.m}")
        if int(a.sum()) <= 0:
            return False
        key = _pattern_bytes(a)
        if key in self.seen:
            return False
        self.seen.add(key)
        if self.P.size == 0:
            self.P = a.reshape(-1, 1).copy()
        else:
            self.P = np.concatenate([self.P, a.reshape(-1, 1)], axis=1)
        return True

    def dedup_P(self) -> None:
        """
        对现有 P 进行去重（一般只在初始化/调试用）
        """
        if self.P.size == 0:
            return
        new_cols = []
        new_seen = set()
        for j in range(self.P.shape[1]):
            a = self.P[:, j].astype(int)
            key = _pattern_bytes(a)
            if key in new_seen:
                continue
            new_seen.add(key)
            new_cols.append(a)
        self.seen = new_seen
        self.P = np.stack(new_cols, axis=1) if new_cols else np.zeros((self.m, 0), dtype=int)