"""
数据 API - 台站列表、时序数据查询、统计信息
读取 data/clean/long.csv
"""
from fastapi import APIRouter, Query
import pandas as pd
from pathlib import Path

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "clean"
LONG_CSV = DATA_DIR / "long.csv"

# 项目主要使用的台站（西昌32井为主模型训练数据）
MAIN_STATIONS = ["SCXC32"]

_df_cache = None

def load_data():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(LONG_CSV, encoding="utf-8-sig")
    return _df_cache


@router.get("/stations")
async def get_stations():
    df = load_data()

    # 只保留项目主台站（西昌32井）
    main_df = df[df["station_id"].isin(MAIN_STATIONS)]

    station_stats = (
        main_df.groupby(["station_id", "station_name"])
        .agg(
            items=("item_name", lambda x: list(x.unique())),
            n_days=("date", "count"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )

    result = []
    for _, row in station_stats.iterrows():
        result.append({
            "id": str(row["station_id"]),
            "name": row["station_name"],
            "items": row["items"],
            "n_days": int(row["n_days"]),
            "date_range": f"{row['date_min']} ~ {row['date_max']}",
            "quality": "usable",
        })
    return result


@router.get("/data/{station_id}")
async def get_timeseries(
    station_id: str,
    item: str = Query(None),
    start: str = Query(None),
    end: str = Query(None),
):
    df = load_data()
    mask = df["station_id"].astype(str) == str(station_id)
    if item:
        mask &= df["item_name"] == item

    filtered = df[mask].sort_values("date").copy()

    if start:
        filtered = filtered[filtered["date"] >= start]
    if end:
        filtered = filtered[filtered["date"] <= end]

    dates = filtered["date"].tolist()
    values = [round(float(v), 4) if pd.notna(v) else None for v in filtered["value"].tolist()]

    return {
        "station_id": station_id,
        "item": item or "all",
        "dates": dates,
        "values": values,
    }


@router.get("/data/{station_id}/stats")
async def get_stats(station_id: str, item: str = Query(None)):
    df = load_data()
    mask = df["station_id"].astype(str) == str(station_id)
    if item:
        mask &= df["item_name"] == item

    filtered = df[mask]
    n_total = len(filtered)
    n_missing = int(filtered["value"].isna().sum())
    missing_ratio = round(n_missing / n_total, 4) if n_total > 0 else 0

    dates = filtered["date"].dropna().tolist()
    date_min = min(dates) if dates else "-"
    date_max = max(dates) if dates else "-"

    items = filtered["item_name"].unique().tolist() if not item else [item]

    return {
        "n_days": n_total,
        "n_missing": n_missing,
        "missing_ratio": missing_ratio,
        "date_range": f"{date_min} ~ {date_max}",
        "items": items,
        "quality": "usable",
    }
