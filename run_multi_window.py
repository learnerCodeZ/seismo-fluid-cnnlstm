"""run_multi_window.py — 批量训练四个窗口（15/30/45/60天）的模型+异常检测。

为 Web 可视化预训练所有窗口的结果。每个窗口产出 summary.json + 预测/异常文件。

用法：python run_multi_window.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.anomaly.lof_detect import lof_scores, detect, k_strategy, window_features
from src.models.evaluate import save_run
from src.models.baselines import naive_predict, fit_rf, predict_rf
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch
from src.models.lstm_rf import fit_lstm_rf, predict_lstm_rf
from src.anomaly.lof_detect import lof_scores, detect, k_strategy
from src.anomaly.events import aggregate_events
from src.anomaly.sensitivity import quantile_sensitivity

SEED = 42
EPOCHS = 200
PATIENCE = 15
TRAIN_RATIO = 0.8
QUANTILE = 0.98
WINDOWS = [15, 30, 45, 60]


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


def run_one_window(window: int, dates, radon, k_train: int, out_dir: Path) -> dict:
    """跑一个窗口大小的完整流程：训练→预测→异常检测→汇总。"""
    set_seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 标准化
    mu, sd = zscore_fit(radon[:k_train])
    rz = zscore_apply(radon, mu, sd)

    # 滑窗
    X, y = make_windows(rz, window)
    tgt_idx = np.arange(window, window + len(y))
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_te, y_te = X[te_m], y[te_m]
    te_dates = dates[tgt_idx[te_m]]

    k_val = int(len(X_tr) * 0.85)

    # CNN-LSTM
    cnn = CNNLSTM(n_vars=1, window=window, conv_channels=(64, 32),
                  lstm_hidden=64, dropout=0.3)
    info = train_torch(cnn, X_tr[..., None], y_tr,
                       X_tr[k_val:, :, None], y_tr[k_val:],
                       epochs=EPOCHS, patience=PATIENCE, seed=SEED)
    pred_cnn = predict_torch(cnn, X_te[..., None])
    m_cnn = save_run(out_dir / "cnn_lstm", "test", y_te, pred_cnn, te_dates,
                     extra={"epochs_run": info["epochs_run"]})

    # LSTM-RF（对照）
    lstm_ext, rf_c, _ = fit_lstm_rf(
        X_tr[..., None], y_tr, X_tr[k_val:, :, None], y_tr[k_val:],
        seed=SEED, epochs=EPOCHS, patience=PATIENCE)
    pred_rf_c = predict_lstm_rf(lstm_ext, rf_c, X_te[..., None])
    m_rf = save_run(out_dir / "lstm_rf", "test", y_te, pred_rf_c, te_dates)

    # 残差 LOF
    residual = y_te - pred_cnn
    feat = window_features(residual, 7)
    valid = ~np.isnan(feat).any(axis=1)
    k = k_strategy(int(valid.sum()), role="residual")
    scores = lof_scores(feat[valid], feat[valid], k)
    fd = pd.DatetimeIndex(te_dates)[6:6 + len(feat)][valid]
    flags, thr = detect(scores, QUANTILE)
    events = aggregate_events(fd, scores, flags)
    sens = quantile_sensitivity(scores)

    # 写文件
    events.to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out_dir / "sensitivity.csv", index=False, encoding="utf-8-sig")

    # summary.json
    summary = {
        "window": window,
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
    dates, radon = load_xichang32()
    k_train = int(len(radon) * TRAIN_RATIO)
    print(f"西昌32井水氡: {len(radon)} 天 | 训练: {k_train} | 测试: {len(radon) - k_train}")
    print(f"窗口: {WINDOWS}\n")

    summaries = {}
    for w in WINDOWS:
        print(f"=== 窗口={w}天 ===")
        out = paths["results_dir"] / f"window_{w}"
        s = run_one_window(w, dates, radon, k_train, out)
        summaries[w] = s
        print(f"  CNN-LSTM R²={s['cnn_lstm']['R2']} | LSTM-RF R²={s['lstm_rf']['R2']}")
        print(f"  异常事件: {s['lof']['n_events']} 个\n")

    # 汇总对比
    print("=" * 70)
    print("四窗口对比")
    print("=" * 70)
    header = f"{'窗口':>6} {'CNN-LSTM R²':>12} {'LSTM-RF R²':>12} {'事件数':>6} {'RMSE':>8}"
    print(header)
    print("-" * len(header))
    for w in WINDOWS:
        s = summaries[w]
        print(f"{w:>5}天 {s['cnn_lstm']['R2']:>12.4f} {s['lstm_rf']['R2']:>12.4f} "
              f"{s['lof']['n_events']:>6} {s['cnn_lstm']['RMSE']:>8.4f}")

    # 写全局汇总
    global_summary = {"windows": summaries}
    (paths["results_dir"] / "window_comparison.json").write_text(
        json.dumps(global_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总: {paths['results_dir'] / 'window_comparison.json'}")


if __name__ == "__main__":
    main()
