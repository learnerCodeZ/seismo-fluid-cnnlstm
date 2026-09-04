"""run_phase4.py — 阶段4 残差LOF异常检测 + 阶段5干扰排除 + 震例对应（西昌32井）。

用 CNN-LSTM 的残差做 LOF 异常检测，然后：
1. 多参数互证（水氡残差 + 原始观测值的异常对比）
2. 干扰排除（与西昌气象三要素对比，排除雨季假异常）
3. 震例对应（异常事件与地震目录的时间对应）

用法：python run_phase4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.anomaly.lof_detect import lof_scores, detect, k_strategy, window_features
from src.anomaly.events import aggregate_events, coincidence
from src.anomaly.sensitivity import quantile_sensitivity
from src.viz.figures import plot_anomaly_scores

SEED = 42
WINDOW = 60
QUANTILE = 0.98
TANGENCY_DAYS = 90  # 地震对应窗口：异常后N天内有地震即算关联


def load_xichang32() -> tuple[pd.DatetimeIndex, np.ndarray]:
    paths = load_paths()
    df = pd.read_csv(paths["clean_dir"] / "long.csv", encoding="utf-8-sig",
                     dtype={"station_id": str, "item_code": str})
    mask = (df["station_id"] == "SCXC32") & (df["item_name"] == "水氡")
    s = df.loc[mask, ["date", "value"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.dropna(subset=["value"]).sort_values("date").drop_duplicates("date")
    s = s.set_index("date")["value"]
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx).interpolate(method="linear", limit=3)
    return s.index.to_numpy(), s.to_numpy()


def load_weather() -> pd.DataFrame:
    """加载西昌气象日值。"""
    paths = load_paths()
    df = pd.read_csv(paths["oridata3_dir"] / "西昌.csv", encoding="gbk")
    # 构建日期
    df["date"] = pd.to_datetime(df["年"].astype(str) + "-" +
                                df["月"].astype(str).str.zfill(2) + "-" +
                                df["日"].astype(str).str.zfill(2))
    df = df.set_index("date")[["气温(℃)", "气压(hpa)", "降水量（mm）"]].copy()
    # 降水单位修正：2010-2019 为0.1mm
    mask_early = df.index.year < 2020
    df.loc[mask_early, "降水量（mm）"] = df.loc[mask_early, "降水量（mm）"] / 10
    # 缺测码
    df[df >= 9999] = pd.NA
    df = df.astype(float)
    return df


def load_earthquake() -> pd.DataFrame:
    paths = load_paths()
    df = pd.read_csv(paths["oridata3_dir"] / "earthquake_catalog.csv", encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    return df


def lof_on_residual(res: np.ndarray, dates: pd.DatetimeIndex,
                     quantile: float = QUANTILE):
    """对残差序列做 LOF（自参照）。"""
    feat = window_features(res, 7)
    valid = ~np.isnan(feat).any(axis=1)
    k = k_strategy(int(valid.sum()), role="residual")
    scores = lof_scores(feat[valid], feat[valid], k)
    d = dates[6:6 + len(feat)][valid]
    flags, thr = detect(scores, quantile)
    return d, scores, flags, thr, k


def check_earthquake_correspondence(events: pd.DataFrame,
                                    eq: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """检查每个异常事件后 TANGENCY_DAYS 天内是否有地震。"""
    rows = []
    for _, ev in events.iterrows():
        ev_end = ev["end"]
        window_end = ev_end + pd.Timedelta(days=TANGENCY_DAYS)
        nearby = eq[(eq["date"] >= ev_end) & (eq["date"] <= window_end)]
        m5 = nearby[nearby["mag"] >= 5.0]
        m4 = nearby[nearby["mag"] >= 4.0]
        rows.append({
            "event_id": ev["event_id"],
            "event_start": ev["start"],
            "event_end": ev["end"],
            "n_points": ev["n_points"],
            "peak_score": round(ev["peak_score"], 3),
            "eq_m4_count": len(m4),
            "eq_m5_count": len(m5),
            "eq_m5_details": "; ".join(
                f"M{r['mag']} {r['date'].strftime('%Y-%m-%d')} {r['place']}"
                for _, r in m5.iterrows()
            ) if len(m5) > 0 else "无",
            "correspondence": "✓" if len(m4) > 0 else "✗",
        })
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(SEED)
    paths = load_paths()
    out_dir = paths["results_dir"] / "phase4_anomaly"

    # ---- 加载数据 ----
    print("加载数据...")
    dates, radon = load_xichang32()
    n = len(radon)
    k_train = int(n * 0.8)
    weather = load_weather()
    eq = load_earthquake()
    print(f"水氡: {n} 天 | 气象: {len(weather)} 天 | 地震: {len(eq)} 事件")

    # ---- 加载 CNN-LSTM 预测结果 ----
    pred_df = pd.read_csv(out_dir.parent / "phase3_cnn_lstm" / "cnn_lstm" / "predictions_test.csv")
    # 阶段3的预测是从第 WINDOW 天开始的，对齐到水氡序列
    pred_dates = pd.to_datetime(pred_df["date"])
    pred_y_true = pred_df["y_true"].values
    pred_y_pred = pred_df["y_pred"].values
    te_residual = pred_y_true - pred_y_pred
    te_dates_arr = pred_dates.values
    print(f"CNN-LSTM 预测: {len(pred_df)} 天 (测试段)")

    # ---- LOF 异常检测 ----
    print("\n[1/4] LOF 异常检测...")
    feat_dates, scores, flags, thr, k = lof_on_residual(te_residual, te_dates_arr)
    print(f"  k={k}, 阈值({QUANTILE*100:.1f}%分位)={thr:.4f}, 异常点={int(flags.sum())}")

    events = aggregate_events(feat_dates, scores, flags)
    print(f"  异常事件: {len(events)} 个")
    if len(events) > 0:
        print(events.to_string(index=False))

    # ---- 残差 vs 原始值异常对比 ----
    print("\n[2/4] 原始观测值 LOF 对照...")
    feat_dates_o, scores_o, flags_o, thr_o, k_o = lof_on_residual(
        pred_y_true, te_dates_arr, QUANTILE)
    events_o = aggregate_events(feat_dates_o, scores_o, flags_o)
    coin = coincidence({"cnn_residual": events, "raw_obs": events_o})
    print("  重合统计:")
    print(coin.to_string(index=False))

    # ---- 敏感性分析 ----
    print("\n[3/4] 敏感性分析...")
    sens = quantile_sensitivity(scores)
    print(sens.to_string(index=False))

    # ---- 气象干扰排除 ----
    print("\n[4/4] 气象干扰排除...")
    events_detail = []
    for _, ev in events.iterrows():
        ev_start = pd.Timestamp(ev["start"])
        ev_end = pd.Timestamp(ev["end"])
        ev_range = pd.date_range(ev_start, ev_end)

        # 对齐到气象数据
        w_sub = weather.reindex(ev_range).dropna(how="all")

        # 判断：事件时段内是否有强降水
        precip_sum = w_sub["降水量（mm）"].sum() if len(w_sub) > 0 else 0
        precip_max = w_sub["降水量（mm）"].max() if len(w_sub) > 0 else 0

        # 判断异常类型：低值还是高值？
        ev_mask = np.isin(dates, ev_range)
        # 残差在该时段的均值（正=观测偏高，负=观测偏低）
        if hasattr(te_residual, '__len__'):
            te_res_series = pd.Series(te_residual, index=pd.DatetimeIndex(te_dates_arr))
        else:
            te_res_series = pd.Series(te_residual, index=dates[k_train + WINDOW:])
        ev_residual_mean = te_res_series.reindex(ev_range).mean()

        # 归因
        attribution = "待定"
        if precip_sum > 10 and ev_residual_mean < 0:
            attribution = "降雨稀释（降水↓水氡↓）"
        elif precip_sum > 10 and ev_residual_mean > 0:
            attribution = "降雨干扰+未知（降水↑但水氡↑，矛盾）"
        else:
            attribution = "非气象因素（潜在构造）"

        events_detail.append({
            "event_id": ev["event_id"],
            "start": ev["start"],
            "end": ev["end"],
            "n_points": ev["n_points"],
            "peak_score": round(ev["peak_score"], 3),
            "residual_direction": "偏高" if ev_residual_mean > 0 else "偏低",
            "precip_sum_mm": round(precip_sum, 1),
            "attribution": attribution,
        })

    detail_df = pd.DataFrame(events_detail)
    print(detail_df.to_string(index=False))

    # ---- 震例对应 ----
    print("\n[5/5] 震例对应...")
    eq_detail = check_earthquake_correspondence(events, eq, pd.DatetimeIndex(feat_dates))
    print(eq_detail.to_string(index=False))

    # ---- 综合判定 ----
    print("\n" + "=" * 65)
    print("综合判定表")
    print("=" * 65)
    final = pd.merge(detail_df, eq_detail[["event_id", "eq_m4_count", "eq_m5_count",
                                            "eq_m5_details", "correspondence"]],
                     on="event_id", how="left")
    final["attribution"] = final["attribution"].fillna("待定")
    print(final.to_string(index=False))

    # 标记：通过全部三道关的异常事件
    passed = final[(final["attribution"].str.contains("潜在构造")) &
                   (final["eq_m4_count"] > 0)]
    print(f"\n通过三道关的事件: {len(passed)}/{len(final)}")
    if len(passed) > 0:
        print("✓ 这些事件排除了气象干扰，且在震前窗口内有对应地震——可能是构造指示异常")
    else:
        print("⚠ 暂无事件同时满足：①非气象因素 ②震前窗口内有对应地震")

    # ---- 落盘 ----
    events.to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(out_dir / "events_with_attribution.csv", index=False, encoding="utf-8-sig")
    eq_detail.to_csv(out_dir / "earthquake_correspondence.csv", index=False, encoding="utf-8-sig")
    final.to_csv(out_dir / "final_verdict.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out_dir / "sensitivity.csv", index=False, encoding="utf-8-sig")
    coin.to_csv(out_dir / "coincidence.csv", index=False, encoding="utf-8-sig")

    # 出图
    plot_anomaly_scores(feat_dates, scores, thr, flags,
                        out_dir / "anomaly_scores.png",
                        title="西昌32井水氡·CNN-LSTM残差·LOF异常得分",
                        extra_events={"地震M5+": pd.DataFrame(
                            [{"start": pd.Timestamp(r["date"])}
                             for _, r in eq[eq["mag"] >= 5].iterrows()
                             if pd.Timestamp(r["date"]).year >= 2024] +
                            [{"start": pd.Timestamp("2026-01-19")}]
                        )})
    print(f"\n结果目录: {out_dir}")


if __name__ == "__main__":
    main()
