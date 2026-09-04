"""run_phase2.py — 阶段2 基线复现（西昌32井水氡，11年长序列）。

四个基线：naive / RF / LSTM / LSTM-RF，与参照文献配置对齐。
健全性检查：排序必须为 RF崩→LSTM降→LSTM-RF好，不符说明管线有bug。

用法：python run_phase2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.models.evaluate import save_run
from src.models.baselines import naive_predict, fit_rf, predict_rf
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch
from src.models.lstm_rf import fit_lstm_rf, predict_lstm_rf
from src.viz.figures import plot_prediction

SEED = 42
WINDOW = 60
EPOCHS = 200
PATIENCE = 15
TRAIN_RATIO = 0.8


def load_xichang32() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """从 long.csv 提取西昌32井水氡日值，按日重索引。"""
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


def main() -> None:
    set_seed(SEED)
    paths = load_paths()
    out_dir = paths["results_dir"] / "phase2_baselines"

    # ---- 加载数据 ----
    dates, radon = load_xichang32()
    n = len(radon)
    k_train = int(n * TRAIN_RATIO)
    print(f"西昌32井水氡: {n} 天 ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})")
    print(f"训练: {k_train} 天 | 测试: {n - k_train} 天")

    # ---- 标准化 ----
    mu, sd = zscore_fit(radon[:k_train])
    rz = zscore_apply(radon, mu, sd)

    # ---- 滑窗 ----
    X, y = make_windows(rz, WINDOW)
    tgt_idx = np.arange(WINDOW, WINDOW + len(y))
    tr_m = tgt_idx < k_train
    te_m = tgt_idx >= k_train
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_te, y_te = X[te_m], y[te_m]
    te_dates = dates[tgt_idx[te_m]]
    print(f"滑窗后: 训练 {len(X_tr)} 样本 | 测试 {len(X_te)} 样本")

    # ---- Naive ----
    print("\n[1/4] naive ...")
    pred_naive = naive_predict(X_te)
    m_naive = save_run(out_dir / "naive", "test", y_te, pred_naive, te_dates)

    # ---- RF ----
    print("[2/4] RF ...")
    rf = fit_rf(X_tr, y_tr, seed=SEED)
    pred_rf = predict_rf(rf, X_te)
    m_rf = save_run(out_dir / "rf", "test", y_te, pred_rf, te_dates)

    # ---- LSTM ----
    print("[3/4] LSTM ...")
    # 验证集取训练段尾部15%（时间顺序）
    k_val = int(len(X_tr) * 0.85)
    lstm_model = CNNLSTM(n_vars=1, window=WINDOW, conv_channels=(1, 1),
                         lstm_hidden=64, dropout=0.3)
    # 用单层LSTM模拟文献结构（CNN通道设为1使其退化为直通）
    info_lstm = train_torch(lstm_model, X_tr[..., None], y_tr,
                            X_tr[k_val:, :, None], y_tr[k_val:],
                            epochs=EPOCHS, patience=PATIENCE, seed=SEED)
    pred_lstm = predict_torch(lstm_model, X_te[..., None])
    m_lstm = save_run(out_dir / "lstm", "test", y_te, pred_lstm, te_dates,
                      extra={"epochs_run": info_lstm["epochs_run"]})

    # ---- LSTM-RF ----
    print("[4/4] LSTM-RF ...")
    lstm_ext, rf_cascade, info_lstmrf = fit_lstm_rf(
        X_tr[..., None], y_tr, X_tr[k_val:, :, None], y_tr[k_val:],
        seed=SEED, epochs=EPOCHS, patience=PATIENCE)
    pred_lstmrf = predict_lstm_rf(lstm_ext, rf_cascade, X_te[..., None])
    m_lstmrf = save_run(out_dir / "lstm_rf", "test", y_te, pred_lstmrf, te_dates,
                        extra={"epochs_run": info_lstmrf["epochs_run"]})

    # ---- 健全性检查 ----
    print("\n" + "=" * 60)
    print("健全性检查：指标排序（参照文献：RF崩→LSTM降→LSTM-RF好）")
    print("=" * 60)
    all_m = {"naive": m_naive, "RF": m_rf, "LSTM": m_lstm, "LSTM-RF": m_lstmrf}
    header = f"{'模型':<12} {'RMSE':>8} {'MAE':>8} {'MAPE(%)':>8} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for name, m in all_m.items():
        print(f"{name:<12} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MAPE']:>8.2f} {m['R2']:>8.4f}")

    # 排序检查
    r2_rank = sorted(all_m.items(), key=lambda x: x[1]["R2"], reverse=True)
    best = r2_rank[0][0]
    worst = r2_rank[-1][0]
    print(f"\nR2 最优: {best} | 最差: {worst}")
    if all_m["RF"]["R2"] < all_m["LSTM"]["R2"] < all_m["LSTM-RF"]["R2"]:
        print("✓ 排序符合预期（RF<LSTM<LSTM-RF），管线正常")
    elif all_m["LSTM-RF"]["R2"] > all_m["RF"]["R2"]:
        print("⚠ LSTM-RF 优于 RF（排序部分符合），但 LSTM 位置需检查")
    else:
        print("✗ 排序异常，需排查管线")

    # ---- 出图 ----
    best_model = r2_rank[0][0].lower().replace("-", "_")
    best_pred = {"naive": pred_naive, "rf": pred_rf, "lstm": pred_lstm,
                 "lstm_rf": pred_lstmrf}[best_model.replace("_rf", "_rf")]
    residual = y_te - best_pred
    fig_path = out_dir / "prediction_best_model.png"
    plot_prediction(y_te, best_pred, residual, te_dates, fig_path,
                    title=f"西昌32井水氡·{best_model}·测试段预测")
    print(f"\n预测对比图: {fig_path}")
    print(f"结果目录: {out_dir}")


if __name__ == "__main__":
    main()
