"""clean.py — 缺测处理与序列级清洗。

规则（与《实施指南》§2.2 一致）：
- 999999（任意前缀长度）识别为缺测 → NaN；
- 每条序列按日重索引（暴露断档）；
- 连续缺失 ≤ max_gap 天线性插值（限序列内部），> max_gap 保留 NaN；
- 行级质量标记：raw / imputed / missing；序列级缺失率统计交给 split.py 的 dataset_info。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_SENTINEL = 999999.0


def mark_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """999999 哨兵值 → NaN。"""
    out = df.copy()
    out.loc[out["value"] >= MISSING_SENTINEL, "value"] = np.nan
    return out


def clean_series(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """对单条序列（同一 station/item/freq/source）完成清洗。

    hourly 序列先聚合为日均值（quality_flag='agg'），再与日值序列同样处理。
    """
    s = df.sort_values("date").drop_duplicates(subset="date").set_index("date")["value"]
    if s.empty:
        return df.assign(value=np.nan, quality_flag="missing")
    base_flag = "raw"
    if df["freq"].iloc[0] == "hourly":
        s = s.resample("D").mean()
        base_flag = "agg"  # 由小时值聚合而来
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    missing_before = s.isna()
    filled = s.interpolate(method="linear", limit=max_gap, limit_area="inside")
    quality = pd.Series(base_flag, index=s.index, dtype=object)
    quality[missing_before & filled.notna()] = "imputed"
    quality[filled.isna()] = "missing"
    out = df.drop_duplicates(subset="date").set_index("date")
    out = out.reindex(full_idx)
    out["value"] = filled
    out["quality_flag"] = quality
    for col in ("source", "station_id", "station_name", "component",
                "item_code", "item_name", "freq"):
        if col in out.columns:
            out[col] = out[col].ffill().bfill()
    return out.reset_index().rename(columns={"index": "date"})


def clean_all(raw: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """对长表按序列分组执行 clean_series。"""
    parts = []
    for _, g in raw.groupby(["station_id", "item_code", "item_name", "freq", "source"],
                            dropna=False):
        parts.append(clean_series(g, max_gap=max_gap))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["station_id", "item_code", "date"]).reset_index(drop=True)
