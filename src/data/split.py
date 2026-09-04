"""split.py — 序列级信息统计与时间顺序训练/测试划分。

产出 dataset_info.csv（即数据质量一览表）：
每条序列一行：天数、起止、缺测/插值统计、质量标记、训练/测试区间。
"""
from __future__ import annotations

import pandas as pd

LOW_QUALITY_MISSING = 0.20  # 缺失率超过该值标记 low_quality


def build_dataset_info(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in clean.groupby(["station_id", "station_name", "item_code",
                                 "item_name", "freq", "source"], dropna=False):
        g = g.sort_values("date")
        n = len(g)
        n_missing = int((g["quality_flag"] == "missing").sum())
        n_imputed = int((g["quality_flag"] == "imputed").sum())
        n_valid = int(g["value"].notna().sum())
        missing_ratio = n_missing / n if n else 1.0
        # 时间顺序 8:2 划分（仅对可用序列记录）
        valid = g.dropna(subset=["value"])
        tr_end = te_start = te_end = pd.NaT
        if len(valid) >= 10:
            k = int(len(valid) * 0.8)
            tr_end = valid["date"].iloc[k - 1]
            te_start = valid["date"].iloc[k]
            te_end = valid["date"].iloc[-1]
        rows.append({
            "station_id": key[0], "station_name": key[1], "item_code": key[2],
            "item_name": key[3], "freq": key[4], "source": key[5],
            "n_days": n, "n_valid": n_valid, "n_missing": n_missing,
            "n_imputed": n_imputed,
            "missing_ratio": round(missing_ratio, 4),
            "quality_flag": "low_quality" if missing_ratio > LOW_QUALITY_MISSING
                            else ("usable" if n_valid >= 10 else "insufficient"),
            "date_start": g["date"].min().date(),
            "date_end": g["date"].max().date(),
            "train_end": tr_end, "test_start": te_start, "test_end": te_end,
        })
    info = pd.DataFrame(rows)
    return info.sort_values(["source", "station_id", "item_name"]).reset_index(drop=True)
