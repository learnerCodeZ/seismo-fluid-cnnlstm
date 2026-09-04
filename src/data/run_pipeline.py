"""run_pipeline.py — 阶段1数据管道总入口（幂等，可重复执行）。

用法：
    python src/data/run_pipeline.py

产出：
    data/clean/long.csv          统一长表（station_id, station_name, item_code,
                                 item_name, freq, source, date, value, quality_flag）
    data/clean/dataset_info.csv  台站×测项质量一览表（含训练/测试划分区间）
    results/pipeline/oridata2_match_log.csv  OriData2 去重匹配日志
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import load_paths, set_seed  # noqa: E402
from src.data import clean as cln  # noqa: E402
from src.data import merge as mrg  # noqa: E402
from src.data import parse as par  # noqa: E402
from src.data import split as spl  # noqa: E402

MAX_GAP = 7  # 连续缺失插值上限（天）


def main() -> None:
    set_seed(42)
    paths = load_paths()

    print("[1/5] parse OriData ...")
    raw_ori = par.parse_oridata(paths["oridata_dir"])
    print(f"      {raw_ori['station_id'].nunique()} stations, "
          f"{len(raw_ori):,} rows from OriData")

    print("[2/5] parse OriData2 ...")
    raw_o2 = par.parse_oridata2(paths["oridata2_dir"])
    print(f"      {len(raw_o2):,} rows from OriData2")

    raw = pd_concat([raw_ori, raw_o2])
    raw = cln.mark_sentinel(raw)

    print("[3/5] match & drop OriData2 duplicates ...")
    kept, match_log = mrg.match_duplicates(raw)
    print(match_log.to_string(index=False))

    print("[4/5] clean (reindex + interpolate) ...")
    clean = cln.clean_all(kept, max_gap=MAX_GAP)
    clean = mrg.merge_station_meta(clean, _station_map(paths))

    print("[5/5] dataset info & save ...")
    info = spl.build_dataset_info(clean)
    match_log_dir = paths["results_dir"] / "pipeline"
    match_log_dir.mkdir(parents=True, exist_ok=True)
    clean_out = clean[["station_id", "station_name", "item_code", "item_name",
                       "freq", "source", "date", "value", "quality_flag"]]
    clean_out.to_csv(paths["clean_dir"] / "long.csv", index=False, encoding="utf-8-sig")
    info.to_csv(paths["clean_dir"] / "dataset_info.csv", index=False, encoding="utf-8-sig")
    match_log.to_csv(match_log_dir / "oridata2_match_log.csv",
                     index=False, encoding="utf-8-sig")

    print("--- summary ---")
    print(f"series: {len(info)}  | usable: {(info['quality_flag'] == 'usable').sum()}  "
          f"| low_quality: {(info['quality_flag'] == 'low_quality').sum()}  "
          f"| insufficient: {(info['quality_flag'] == 'insufficient').sum()}")
    print(f"saved: {paths['clean_dir'] / 'long.csv'}")
    print(f"saved: {paths['clean_dir'] / 'dataset_info.csv'}")


def _station_map(paths) -> dict:
    return par.parse_station_meta(paths["station_meta"])[0]


def pd_concat(frames):
    import pandas as pd
    cols = ["source", "station_id", "station_name", "component", "item_code",
            "item_name", "freq", "date", "value"]
    str_cols = ["source", "station_id", "station_name", "component",
                "item_code", "item_name", "freq"]
    aligned = []
    for f in frames:
        f = f.copy()
        for c in cols:
            if c not in f.columns:
                f[c] = ""
        for c in str_cols:  # 空值填空串，防止 CSV 读回时被推断为数值类型
            f[c] = f[c].fillna("").astype(str)
        aligned.append(f[cols])
    return pd.concat(aligned, ignore_index=True)


if __name__ == "__main__":
    main()
