"""anomaly/lof_detect.py — 残差 LOF 异常检测（方法对齐参照文献）。

核心逻辑：最优模型的训练段残差应为白噪声；把训练段残差特征作为 LOF 的
"正常密度参照"（novelty=True），对全序列（重点是测试段）打分，
得分超过高分位数阈值者判为候选异常点。
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

ROLE_K_RANGE = {          # 文献思路：不同检测对象用不同的 k 邻域范围
    "residual": (5, 25),  # 残差：小 k，抓细微异常
    "raw": (20, 60),      # 原始序列：大 k，抓宏观异常
    "medium": (10, 30),   # 水位/水温等互证参数：中等 k
}


def k_strategy(n: int, role: str = "residual") -> int:
    """k 值动态选取：基准 √n，界于样本量 1%~10%，再落入角色区间。"""
    lo_n, hi_n = max(2, int(np.ceil(0.01 * n))), max(3, int(np.floor(0.10 * n)))
    lo, hi = ROLE_K_RANGE.get(role, (5, 25))
    lo, hi = max(lo, lo_n), min(hi, max(hi, lo_n))
    k = int(round(np.sqrt(n)))
    return int(np.clip(k, lo, hi))


def lof_scores(ref_x: np.ndarray, score_x: np.ndarray, k: int) -> np.ndarray:
    """用参照特征拟合 LOF（novelty），对 score_x 打分；得分越大越异常。

    ref_x / score_x: [N, d] 特征矩阵（残差原始值或滑窗特征）。
    """
    lof = LocalOutlierFactor(n_neighbors=min(k, len(ref_x) - 1), novelty=True)
    lof.fit(np.asarray(ref_x, float))
    return -lof.score_samples(np.asarray(score_x, float))  # sklearn: 越小越异常 → 取负


def detect(scores: np.ndarray, quantile: float = 0.987) -> tuple[np.ndarray, float]:
    """分位数阈值判定。返回 (异常布尔数组, 阈值)。"""
    thr = float(np.quantile(scores, quantile))
    return scores > thr, thr


def window_features(x: np.ndarray, window: int = 7) -> np.ndarray:
    """把一维残差序列转成 [N, window] 特征（局部形态），末尾不足处丢弃。"""
    x = np.asarray(x, float)
    n = len(x) - window + 1
    idx = np.arange(window) + np.arange(n)[:, None]
    return x[idx]
