"""run_multi_var.py — 多变量批量训练（水氡/水位/双变量/多变量×6窗口）。

为 Web 可视化预训练所有变量组合的结果。

用法：python run_multi_var.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.features.build import make_windows, make_windows_multi, zscore_apply, zscore_fit
from src.models.evaluate import save_run
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch
from src.models.lstm_rf import fit_lstm_rf, predict_lstm_rf
from src.anomaly.lof_detect import lof_scores, detect, k_strategy, window_features
from src.anomaly.events import aggregate_events
from src.anomaly.sensitivity import quantile_sensitivity

SEED = 42
EPOCHS = 200
PATIENCE = 15
TRAIN_RATIO = 0.8
QUANTILE = 0.98
WINDOWS = [15, 30, 45, 60, 90, 120]


def load_xichang32_radon() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """加载西昌32井水氡日值。"""
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


def load_xichang32_level() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """加载西昌32井动水位（小时值→日均值）。"""
    paths = load_paths()
    level_file = paths["oridata3_dir"] / "西昌川32井动水位整点值.TXT"
    # 跳过前4行文件头
    df = pd.read_csv(level_file, sep=' ', header=None, skiprows=4,
                     names=['datetime', 'value'], encoding='gbk', dtype=str)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H', errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['datetime', 'value'])
    # 按日聚合为均值
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date')['value'].mean()
    daily.index = pd.to_datetime(daily.index)
    # 与水氡时间对齐
    paths2 = load_paths()
    radon_df = pd.read_csv(paths2["clean_dir"] / "long.csv", encoding="utf-8-sig",
                           dtype={"station_id": str, "item_code": str})
    radon_mask = (radon_df["station_id"] == "SCXC32") & (radon_df["item_name"] == "水氡")
    radon_dates = pd.to_datetime(radon_df.loc[radon_mask, "date"]).sort_values().unique()
    full_idx = pd.DatetimeIndex(radon_dates)
    daily = daily.reindex(full_idx).interpolate(method="linear", limit=3)
    return daily.index.to_numpy(), daily.to_numpy()


def load_weather() -> pd.DataFrame:
    """加载西昌气象日值（降水、气温、气压）。"""
    paths = load_paths()
    df = pd.read_csv(paths["oridata3_dir"] / "西昌.csv", encoding="gbk")
    df["date"] = pd.to_datetime(df["年"].astype(str) + "-" +
                                df["月"].astype(str).str.zfill(2) + "-" +
                                df["日"].astype(str).str.zfill(2))
    df = df.set_index("date")[["气温(℃)", "气压(hpa)", "降水量（mm）"]].copy()
    # 降水单位修正
    mask_early = df.index.year < 2020
    df.loc[mask_early, "降水量（mm）"] = df.loc[mask_early, "降水量（mm）"] / 10
    df[df >= 9999] = pd.NA
    df = df.astype(float)
    # 与水氡时间对齐
    radon_df = pd.read_csv(paths["clean_dir"] / "long.csv", encoding="utf-8-sig",
                           dtype={"station_id": str, "item_code": str})
    radon_mask = (radon_df["station_id"] == "SCXC32") & (radon_df["item_name"] == "水氡")
    radon_dates = pd.to_datetime(radon_df.loc[radon_mask, "date"]).sort_values().unique()
    df = df.reindex(pd.DatetimeIndex(radon_dates)).interpolate(method="linear", limit=3)
    return df


def run_one_config(var_name: str, window: int, X: np.ndarray, y: np.ndarray,
                   dates: np.ndarray, k_train: int, out_dir: Path) -> dict:
    """跑一个变量组合×窗口的完整流程。"""
    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    # X 是 [samples, n_vars] 的2D数组
    n_vars = X.shape[1] if X.ndim == 2 else 1
    if n_vars == 1:
        X_win, y_win = make_windows(X, window)
    else:
        X_win, y_win = make_windows_multi(X, window, target_col=0)

    tgt_idx = np.arange(window, window + len(y_win))
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    X_tr, y_tr = X_win[tr_m], y_win[tr_m]
    X_te, y_te = X_win[te_m], y_win[te_m]
    te_dates = dates[tgt_idx[te_m]]

    k_val = int(len(X_tr) * 0.85)

    # CNN-LSTM
    cnn = CNNLSTM(n_vars=n_vars, window=window, conv_channels=(64, 32),
                  lstm_hidden=64, dropout=0.3)
    X_tr_3d = X_tr if X_tr.ndim == 3 else X_tr[..., None]
    X_te_3d = X_te if X_te.ndim == 3 else X_te[..., None]
    X_val_3d = X_tr_3d[k_val:]
    y_val = y_tr[k_val:]
    info = train_torch(cnn, X_tr_3d, y_tr, X_val_3d, y_val,
                       epochs=EPOCHS, patience=PATIENCE, seed=SEED)
    pred_cnn = predict_torch(cnn, X_te_3d)
    m_cnn = save_run(out_dir / "cnn_lstm", "test", y_te, pred_cnn, te_dates,
                     extra={"epochs_run": info["epochs_run"]})

    # LSTM-RF
    lstm_ext, rf_c, _ = fit_lstm_rf(X_tr_3d, y_tr, X_val_3d, y_val,
                                     seed=SEED, epochs=EPOCHS, patience=PATIENCE)
    pred_rf = predict_lstm_rf(lstm_ext, rf_c, X_te_3d)
    m_rf = save_run(out_dir / "lstm_rf", "test", y_te, pred_rf, te_dates)

    # 残差 LOF
    residual = y_te - pred_cnn
    feat = window_features(residual, 7)
    valid = ~np.isnan(feat).any(axis=1)
    k = k_strategy(int(valid.sum()), role="residual")
    scores = lof_scores(feat[valid], feat[valid], k)
    fd = pd.DatetimeIndex(te_dates)[6:6 + len(feat)][valid]
    flags, thr = detect(scores, QUANTILE)
    events = aggregate_events(fd, scores, flags)

    events.to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")
    sens = quantile_sensitivity(scores)
    sens.to_csv(out_dir / "sensitivity.csv", index=False, encoding="utf-8-sig")

    summary = {
        "var_name": var_name,
        "window": window,
        "n_vars": n_vars,
        "n_train": int(tr_m.sum()),
        "n_test": int(te_m.sum()),
        "cnn_lstm": {"RMSE": round(m_cnn["RMSE"], 4), "MAE": round(m_cnn["MAE"], 4),
                     "MAPE": round(m_cnn["MAPE"], 2), "R2": round(m_cnn["R2"], 4)},
        "lstm_rf": {"RMSE": round(m_rf["RMSE"], 4), "MAE": round(m_rf["MAE"], 4),
                    "MAPE": round(m_rf["MAPE"], 2), "R2": round(m_rf["R2"], 4)},
        "lof": {"k": k, "threshold": round(thr, 4), "n_anomalies": int(flags.sum()),
                "n_events": len(events)},
        "events": [{"event_id": int(r["event_id"]),
                    "start": str(r["start"].date()),
                    "end": str(r["end"].date()),
                    "n_points": int(r["n_points"]),
                    "peak_score": round(float(r["peak_score"]), 3)}
                   for _, r in events.iterrows()],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return summary


def main() -> None:
    set_seed(SEED)
    paths = load_paths()

    print("加载数据...")
    dates_r, radon = load_xichang32_radon()
    dates_l, level = load_xichang32_level()
    weather = load_weather()
    k_train = int(len(radon) * TRAIN_RATIO)

    # 标准化
    mu_r, sd_r = zscore_fit(radon[:k_train])
    radon_z = zscore_apply(radon, mu_r, sd_r)
    mu_l, sd_l = zscore_fit(level[:k_train])
    level_z = zscore_apply(level, mu_l, sd_l)
    temp_z = zscore_apply(weather["气温(℃)"].values, *zscore_fit(weather["气温(℃)"].values[:k_train]))
    pres_z = zscore_apply(weather["气压(hpa)"].values, *zscore_fit(weather["气压(hpa)"].values[:k_train]))
    prec_z = zscore_apply(weather["降水量（mm）"].values, *zscore_fit(weather["降水量（mm）"].values[:k_train]))

    # 变量组合
    configs = {
        "radon": {"X": radon_z, "y": radon_z, "dates": dates_r},
        "level": {"X": level_z, "y": level_z, "dates": dates_l},
        "radon_level": {"X": np.c_[radon_z, level_z], "y": radon_z, "dates": dates_r},
        "radon_level_weather": {"X": np.c_[radon_z, level_z, prec_z, temp_z, pres_z],
                                "y": radon_z, "dates": dates_r},
    }

    all_summaries = {}
    for var_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"变量组合: {var_name}")
        print(f"{'='*60}")
        for w in WINDOWS:
            out = paths["results_dir"] / var_name / f"window_{w}"
            s = run_one_config(var_name, w, cfg["X"], cfg["y"], cfg["dates"], k_train, out)
            all_summaries[f"{var_name}_{w}"] = s
            print(f"  窗口={w:3d}天 | CNN-LSTM R²={s['cnn_lstm']['R2']:.4f} | LSTM-RF R²={s['lstm_rf']['R2']:.4f} | 事件={s['lof']['n_events']}")

    # 全局汇总
    print(f"\n{'='*80}")
    print("全局对比")
    print(f"{'='*80}")
    header = f"{'变量组合':<25} {'窗口':>4} {'CNN-LSTM R²':>12} {'LSTM-RF R²':>12} {'事件数':>6}"
    print(header)
    print("-" * len(header))
    for var_name in configs:
        for w in WINDOWS:
            key = f"{var_name}_{w}"
            s = all_summaries[key]
            print(f"{var_name:<25} {w:>3}天 {s['cnn_lstm']['R2']:>12.4f} {s['lstm_rf']['R2']:>12.4f} {s['lof']['n_events']:>6}")

    (paths["results_dir"] / "multi_var_comparison.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总: {paths['results_dir'] / 'multi_var_comparison.json'}")


if __name__ == "__main__":
    main()
