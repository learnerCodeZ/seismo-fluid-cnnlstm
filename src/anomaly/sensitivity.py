"""anomaly/sensitivity.py — k × 分位数阈值敏感性分析。

结论报告必须附稳定性结果（见《研究计划书》工程约定）：
对每个 (k, quantile) 组合输出异常点集合，统计事件数与 Jaccard 稳定性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.98, 0.985, 0.987, 0.99, 0.995)


def quantile_sensitivity(scores: np.ndarray, dates=None,
                         quantiles=DEFAULT_QUANTILES) -> pd.DataFrame:
    """固定 k，扫分位数：报告每个阈值下的异常点数（供稳定性对照）。"""
    scores = np.asarray(scores)
    rows = []
    for q in quantiles:
        thr = float(np.quantile(scores, q))
        n = int((scores > thr).sum())
        rows.append({"quantile": q, "threshold": round(thr, 5), "n_anomalies": n})
    return pd.DataFrame(rows)


def k_sensitivity(ref_x: np.ndarray, score_x: np.ndarray, k_list: list[int],
                  quantile: float = 0.987) -> pd.DataFrame:
    """固定分位数，扫 k：不同 k 下异常点集合的 Jaccard 相似度（以中位 k 为参照）。"""
    from src.anomaly.lof_detect import lof_scores, detect

    sets = {}
    for k in k_list:
        sc = lof_scores(ref_x, score_x, k)
        flags, _ = detect(sc, quantile)
        sets[k] = set(np.where(flags)[0].tolist())
    ref_k = k_list[len(k_list) // 2]
    ref_set = sets[ref_k]
    rows = []
    for k in k_list:
        inter = len(sets[k] & ref_set)
        union = len(sets[k] | ref_set) or 1
        rows.append({"k": k, "n_anomalies": len(sets[k]),
                     "jaccard_vs_median_k": round(inter / union, 3)})
    return pd.DataFrame(rows)
