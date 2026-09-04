"""merge.py — 台站元数据合并与 OriData2 去重。

- OriData 台站名来自元数据表（parse 阶段已映射）；
- OriData2 的序列通过"数值匹配"寻找 OriData 中的同源序列：重叠 ≥60 天且
  相对 RMSE（RMSE/参照均值）< 5% 判定为重复 → 丢弃并记录匹配结果；
  未匹配上的 OriData2 序列（如西昌32井，四川台站，不在云南元数据内）保留。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATCH_MIN_OVERLAP = 60
MATCH_REL_RMSE = 0.05


def _pivot_daily(df: pd.DataFrame) -> dict[tuple, pd.Series]:
    """把日值长表转为 {(station_id, item_name): value_series}。"""
    daily = df[df["freq"] == "daily"]
    out = {}
    for key, g in daily.groupby(["station_id", "item_name"], dropna=False):
        s = g.drop_duplicates("date").set_index("date")["value"]
        out[key] = s
    return out


def _item_group(item_name: str) -> str:
    """测项大类：氡 / 水位 / 其他。跨大类禁止匹配（水温与水氡数值区间可能重叠）。"""
    if "氡" in item_name:
        return "radon"
    if "水位" in item_name:
        return "level"
    return "other"


def match_duplicates(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OriData2 与 OriData 数值匹配去重（仅在同测项大类内匹配）。

    返回 (kept_raw, match_log)：kept_raw 为去除重复后的原始长表；
    match_log 记录每个 OriData2 序列的匹配结果。
    """
    daily = raw[raw["freq"] == "daily"].copy()
    ref = _pivot_daily(daily[daily["source"] == "oridata"])
    log, drop_idx = [], []
    for key, g in daily[daily["source"] == "oridata2"].groupby(
            ["station_id", "item_name"], dropna=False):
        s2 = g.drop_duplicates("date").set_index("date")["value"]
        group = _item_group(key[1])
        best_key, best_rrmse = None, np.inf
        for rkey, s1 in ref.items():
            if _item_group(rkey[1]) != group:
                continue  # 跨测项（如水温 vs 水氡）不做数值匹配
            idx = s2.index.intersection(s1.index)
            if len(idx) < MATCH_MIN_OVERLAP:
                continue
            a, b = s1.loc[idx], s2.loc[idx]
            ok = a.notna() & b.notna()
            if ok.sum() < MATCH_MIN_OVERLAP:
                continue
            a, b = a[ok], b[ok]
            mean_ref = a.abs().mean()
            rrmse = np.sqrt(np.mean((a - b) ** 2)) / mean_ref if mean_ref else np.inf
            if rrmse < best_rrmse:
                best_key, best_rrmse = rkey, rrmse
        if best_key is not None and best_rrmse < MATCH_REL_RMSE:
            drop_idx.extend(g.index)
            verdict = "duplicate"
        else:
            verdict = "keep"
        log.append({"oridata2_station": key[0], "oridata2_item": key[1],
                    "n_days": len(s2), "matched_to": str(best_key),
                    "rel_rmse": round(best_rrmse, 5) if np.isfinite(best_rrmse) else None,
                    "verdict": verdict})
    kept = raw.drop(index=drop_idx)
    return kept, pd.DataFrame(log)


def merge_station_meta(clean: pd.DataFrame, station_map: dict) -> pd.DataFrame:
    """为 OriData 序列补全台站名（OriData2 已带中文名）。"""
    out = clean.copy()
    mask = out["source"] == "oridata"
    out.loc[mask, "station_name"] = out.loc[mask, "station_id"].map(station_map)
    out["station_name"] = out["station_name"].fillna(out["station_id"])
    return out
