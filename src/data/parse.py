"""parse.py — 原始数据解析。

三个来源统一解析为"长表中间格式"（含 source/freq 元信息，值保持原样，999999 尚不处理）：
1. OriData/*.TXT      ：GBK，两列 YYYYMMDD value，文件名 {台站}_{分量}_{测项码}_{日|时}.TXT
2. B区域_云南省.xlsx   ：单列 CSV 文本（需拆分+去引号），提供台站名与测项名映射
3. OriData2/**/*.xlsx ：双列表，日期格式混杂（8 位/10 位），表头名不统一
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd

RAW_COLS = ["source", "station_id", "component", "item_code", "item_name",
            "freq", "date", "value"]


def _clean_meta_cell(v) -> str:
    """元数据单元格去空格与引号。"""
    return v.strip().strip('"').strip() if isinstance(v, str) else ""


def parse_station_meta(meta_path: Path) -> tuple[dict, dict]:
    """解析 B区域_云南省.xlsx。

    返回 (station_map, item_map)：
    - station_map: 台站代码 -> 台站名
    - item_map:    分量编码 -> 测项名称
    """
    raw = pd.read_excel(meta_path, header=None, dtype=str)
    rows = []
    for v in raw.iloc[:, 0]:
        if isinstance(v, str) and "," in v:
            rows.append([_clean_meta_cell(x) for x in v.split(",")])
    wide = pd.DataFrame(rows)
    station_map = dict(zip(wide[1], wide[2]))
    item_map = dict(zip(wide[10], wide[12]))
    return station_map, item_map


def _parse_date_column(s: pd.Series) -> pd.Series:
    """兼容 8 位（YYYYMMDD）与 10 位（YYYYMMDDHH）日期字符串。"""
    s = s.astype(str).str.strip().str.replace("-", "", regex=False)
    out = pd.to_datetime(s.where(s.str.len() == 8), format="%Y%m%d", errors="coerce")
    hourly = pd.to_datetime(s.where(s.str.len() == 10), format="%Y%m%d%H", errors="coerce")
    return out.fillna(hourly)


def _read_value_file(path: Path, source: str, station_id: str, item_code: str,
                     item_name: str) -> pd.DataFrame:
    """读取两列（时间, 数值）数据文件（TXT 或 xlsx 通用）。"""
    if path.suffix.lower() == ".txt":
        with open(path, encoding="gbk", errors="replace") as f:
            recs = []
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    recs.append((parts[0], parts[1]))
        df = pd.DataFrame(recs, columns=["date", "value"])
    else:
        df = pd.read_excel(path, header=0, dtype=str)
        df.columns = ["date", "value"][: len(df.columns)]
    df["date"] = _parse_date_column(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    freq = "hourly" if df["date"].dt.minute.eq(0).all() and (
        df["date"].diff().dropna().dt.total_seconds().median() <= 3600) else "daily"
    df["freq"] = freq
    df["source"] = source
    df["station_id"] = station_id
    df["component"] = ""
    df["item_code"] = item_code
    df["item_name"] = item_name
    return df[RAW_COLS]


def parse_oridata(oridata_dir: Path) -> pd.DataFrame:
    """解析 OriData 全部 TXT 日值/小时值文件。"""
    station_map, item_map = parse_station_meta(oridata_dir / "B区域_云南省.xlsx")
    frames = []
    for f in sorted(glob.glob(str(oridata_dir / "*.TXT"))):
        base = Path(f).stem  # e.g. 53034_2_4112_日 / 53123_C_4222_时
        parts = base.split("_")
        if len(parts) != 4:
            continue
        st, comp, item, freq_cn = parts
        freq = "daily" if freq_cn == "日" else "hourly"
        dates, values = [], []
        with open(f, encoding="gbk", errors="replace") as fh:
            lines = [line.split() for line in fh if line.strip()]
        if freq == "hourly" and lines and len(lines[0]) > 2:
            # 宽表格式：每行 = 日期 + 24 个小时值 → 熔融为长表
            recs = []
            for toks in lines:
                for h, v in enumerate(toks[1:24]):
                    recs.append((f"{toks[0]}{h:02d}", v))
            df = pd.DataFrame(recs, columns=["date", "value"])
        else:
            df = pd.DataFrame([(t[0], t[1]) for t in lines if len(t) == 2],
                              columns=["date", "value"])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H" if freq == "hourly"
                                    else "%Y%m%d", errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["freq"] = freq
        df["source"] = "oridata"
        df["station_id"] = st
        df["component"] = comp
        df["item_code"] = item
        df["item_name"] = item_map.get(item, item)
        frames.append(df[RAW_COLS])
    out = pd.concat(frames, ignore_index=True)
    out["station_name"] = out["station_id"].map(station_map).fillna(out["station_id"])
    return out


def parse_oridata2(oridata2_dir: Path) -> pd.DataFrame:
    """解析 OriData2 全部 xlsx；台站/测项从文件名与所在子目录推断。"""
    frames = []
    for f in sorted(glob.glob(str(oridata2_dir / "**" / "*.xlsx"), recursive=True)):
        stem = Path(f).stem
        if "氡气" in stem:
            item_name = "气氡"
        elif "水氡" in stem:
            item_name = "水氡"
        elif "水位" in stem:
            item_name = "水位"
        elif "气氡数值" in stem:
            item_name = "气氡"
        else:
            item_name = "水氡"  # 水氡数据8.21 子目录内无后缀文件默认为水氡
        # 台站名：剥离测项词尾（氡气/水氡值/水氡/水位/气氡数值/值/数值）
        st = re.sub(r"(氡气数值|气氡数值|水氡值|氡气|水氡|气氡|水位|数值|值)$", "", stem)
        if "西昌32井" in stem:
            st_id = "SCXC32"
        else:
            st_id = st
        df = _read_value_file(Path(f), "oridata2", st_id, "", item_name)
        df["station_name"] = st
        frames.append(df[RAW_COLS + ["station_name"]])
    out = pd.concat(frames, ignore_index=True)
    return out
