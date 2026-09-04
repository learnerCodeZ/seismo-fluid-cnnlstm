"""run_phase3.py — 阶段3 CNN-LSTM 单变量训练（西昌32井水氡）。

在阶段2基线的基础上加入CNN-LSTM主模型，与LSTM-RF对比。
CNN的卷积层负责提取局部突变特征，LSTM负责时序依赖理解。

用法：python run_phase3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.features.build import make_windows, zscore_apply, zscore_fit
from src.models.evaluate import save_run, all_metrics
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
    """从 long.csv 提取西昌32井水氡日值。"""
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
    out_dir = paths["results_dir"] / "phase3_cnn_lstm"

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

    # ---- 验证集（训练段尾部15%，时间顺序）----
    k_val = int(len(X_tr) * 0.85)

    # ---- CNN-LSTM（主模型）----
    print("\n[CNN-LSTM] 训练中...")
    cnn_model = CNNLSTM(n_vars=1, window=WINDOW,
                        conv_channels=(64, 32), lstm_hidden=64, dropout=0.3)
    info_cnn = train_torch(cnn_model, X_tr[..., None], y_tr,
                           X_tr[k_val:, :, None], y_tr[k_val:],
                           epochs=EPOCHS, patience=PATIENCE, seed=SEED)
    pred_cnn = predict_torch(cnn_model, X_te[..., None])
    m_cnn = save_run(out_dir / "cnn_lstm", "test", y_te, pred_cnn, te_dates,
                     extra={"epochs_run": info_cnn["epochs_run"]})

    # ---- LSTM-RF（文献最优，作为对照）----
    print("[LSTM-RF] 训练中...")
    lstm_ext, rf_cascade, info_lstmrf = fit_lstm_rf(
        X_tr[..., None], y_tr, X_tr[k_val:, :, None], y_tr[k_val:],
        seed=SEED, epochs=EPOCHS, patience=PATIENCE)
    pred_lstmrf = predict_lstm_rf(lstm_ext, rf_cascade, X_te[..., None])
    m_lstmrf = save_run(out_dir / "lstm_rf", "test", y_te, pred_lstmrf, te_dates,
                        extra={"epochs_run": info_lstmrf["epochs_run"]})

    # ---- RF（文献配置，作为下界对照）----
    print("[RF] 训练中...")
    rf = fit_rf(X_tr, y_tr, seed=SEED)
    pred_rf = predict_rf(rf, X_te)
    m_rf = save_run(out_dir / "rf", "test", y_te, pred_rf, te_dates)

    # ---- 对比 ----
    print("\n" + "=" * 65)
    print("阶段3 结果：CNN-LSTM vs LSTM-RF vs RF")
    print("=" * 65)
    all_m = {"RF": m_rf, "LSTM-RF": m_lstmrf, "CNN-LSTM": m_cnn}
    header = f"{'模型':<12} {'RMSE':>8} {'MAE':>8} {'MAPE(%)':>8} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for name, m in all_m.items():
        print(f"{name:<12} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MAPE']:>8.2f} {m['R2']:>8.4f}")

    # 胜负判定
    if m_cnn["R2"] > m_lstmrf["R2"]:
        print(f"\n✓ CNN-LSTM({m_cnn['R2']:.4f}) > LSTM-RF({m_lstmrf['R2']:.4f})")
        print("  卷积层的局部特征提取带来了增益！")
    elif m_cnn["R2"] > m_rf["R2"]:
        print(f"\n✓ CNN-LSTM({m_cnn['R2']:.4f}) > RF({m_rf['R2']:.4f})")
        print(f"  但未超过 LSTM-RF({m_lstmrf['R2']:.4f})，可能需要调参或增加训练数据")
    else:
        print(f"\n✗ CNN-LSTM({m_cnn['R2']:.4f}) 不如 RF({m_rf['R2']:.4f})，需排查")

    # ---- 出图 ----
    fig_path = out_dir / "cnn_lstm_vs_lstm_rf.png"
    plot_prediction(y_te, pred_cnn, y_te - pred_cnn, te_dates, fig_path,
                    title="西昌32井水氡·CNN-LSTM·测试段预测")
    print(f"\n预测对比图: {fig_path}")

    # 对比图：CNN-LSTM vs LSTM-RF
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(te_dates, y_te, lw=0.8, alpha=0.6, label="观测值", color="gray")
    a1.plot(te_dates, pred_lstmrf, lw=0.8, alpha=0.8, label=f"LSTM-RF (R²={m_lstmrf['R2']:.4f})")
    a1.plot(te_dates, pred_cnn, lw=0.8, alpha=0.8, label=f"CNN-LSTM (R²={m_cnn['R2']:.4f})")
    a1.set_ylabel("value")
    a1.legend(loc="upper right")
    a1.set_title("西昌32井水氡·CNN-LSTM vs LSTM-RF")
    a1.spines["top"].set_visible(False)
    a1.spines["right"].set_visible(False)

    a2.plot(te_dates, y_te - pred_lstmrf, lw=0.7, alpha=0.8, label="LSTM-RF 残差")
    a2.plot(te_dates, y_te - pred_cnn, lw=0.7, alpha=0.8, label="CNN-LSTM 残差")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("residual")
    a2.legend(loc="upper right", fontsize=8)
    a2.spines["top"].set_visible(False)
    a2.spines["right"].set_visible(False)
    a2.grid(True, ls="--", lw=0.5, alpha=0.3)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    cmp_path = out_dir / "cnn_vs_lstmrf_comparison.png"
    fig.savefig(cmp_path, dpi=300)
    plt.close(fig)
    print(f"对比图: {cmp_path}")
    print(f"结果目录: {out_dir}")


if __name__ == "__main__":
    main()
