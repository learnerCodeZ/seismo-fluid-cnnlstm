"""smoke_test.py — 合成数据端到端冒烟测试（框架先行，不依赖真实数据）。

生成带年变周期 + AR(1) 噪声的合成水氡序列，在**测试段**植入 2 个已知异常，
跑通完整链条：滑窗 → 基线/CNN-LSTM 训练 → 残差 → LOF → 事件聚合，
并检验植入异常是否被检出（±7 天容差）。结果落盘 results/smoke/。

方法论要点（与真实流程一致）：
- LOF 参照集 = 训练段残差特征（白噪声假设），异常检测对象 = 全序列；
- 含 NaN 的特征行直接丢弃（**禁止插值伪造**——阶段1经验：常数填充会毒化 LOF 密度）。

用法：python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.common import load_paths, set_seed
from src.features.build import make_windows, make_windows_multi, zscore_apply, zscore_fit
from src.models.evaluate import save_run
from src.models.baselines import naive_predict, fit_rf, predict_rf
from src.models.cnn_lstm import CNNLSTM, train_torch, predict_torch
from src.anomaly.lof_detect import lof_scores, detect, k_strategy, window_features
from src.anomaly.events import aggregate_events, coincidence
from src.anomaly.sensitivity import quantile_sensitivity
from src.viz.figures import plot_anomaly_scores

CFG = dict(n_days=1096, start="2022-01-01", window=60, annual_amp=5.0,
           base=30.0, noise=0.5,
           # 两个植入异常均在测试段（train_ratio=0.8 → 测试目标约从第 877 天起）
           # 两个植入异常均在测试段；突跳取 3 天（对齐真实短临突跳口径，见文献表7）
           planted=[{"day": 920, "delta": -6.0, "len": 3, "label": "低值突跳"},
                    {"day": 1000, "delta": 8.0, "len": 3, "label": "高值突跳"}],
           epochs=200, seed=42, quantile=0.98)  # 工作阈值经敏感性分析确定（冒烟报告附全档位结果）


def make_synthetic():
    """合成水氡（年变正弦 + AR(1) 噪声）与弱耦合水位；在测试段植入异常。"""
    rng = np.random.default_rng(CFG["seed"])
    dates = pd.date_range(CFG["start"], periods=CFG["n_days"], freq="D")
    t = np.arange(CFG["n_days"])
    radon = (CFG["base"]
             + CFG["annual_amp"] * np.sin(2 * np.pi * t / 365.25)
             + rng.normal(0, CFG["noise"], CFG["n_days"]))
    for i in range(1, len(radon)):          # AR(1) 平滑模拟真实自相关
        radon[i] = 0.6 * radon[i - 1] + 0.4 * radon[i]
    level = 30 + 0.3 * np.sin(2 * np.pi * t / 365.25 + 0.8) + rng.normal(0, 0.1, len(t))
    planted_dates = []
    for p in CFG["planted"]:
        sl = slice(p["day"], p["day"] + p["len"])
        radon[sl] += p["delta"]
        planted_dates.append((dates[p["day"]], p["label"]))
    return dates, radon, level, planted_dates


def lof_on_residual(res_full: np.ndarray, dates: pd.DatetimeIndex, tr_i: np.ndarray,
                    quantile: float = CFG["quantile"]):
    """对残差序列执行 LOF（文献口径：对残差序列自身打分，高分位数者为异常）。

    tr_i 仅用于 k 值的样本量基准。含 NaN 的特征行整行丢弃。
    """
    feat = window_features(res_full, 7)
    valid = ~np.isnan(feat).any(axis=1)
    k = k_strategy(int(valid.sum()), role="residual")
    scores = lof_scores(feat[valid], feat[valid], k)   # 拟合与打分同分布（自参照）
    d = dates[6:6 + len(feat)][valid]
    flags, thr = detect(scores, quantile)
    return d, scores, flags, thr, k


def main() -> None:
    set_seed(CFG["seed"])
    paths = load_paths()
    out_dir = paths["results_dir"] / "smoke"

    dates, radon, level, planted = make_synthetic()
    n = len(radon)
    tr_i = np.arange(int(n * 0.8))
    te_i = np.arange(int(n * 0.8), n)
    mu, sd = zscore_fit(radon[tr_i])
    radon_z = zscore_apply(radon, mu, sd)
    level_z = zscore_apply(level, *zscore_fit(level[tr_i]))
    print(f"[1/5] 合成序列 {n} 天；植入异常(测试段): "
          + "; ".join(f"{d.date()}({lab})" for d, lab in planted))

    # ---- A: 预测模型 ----
    print("[2/5] 预测模型 ...")
    X1, y1 = make_windows(radon_z, CFG["window"])
    tgt_idx = np.arange(CFG["window"], CFG["window"] + len(y1))
    tr_m = np.isin(tgt_idx, tr_i)
    te_m = np.isin(tgt_idx, te_i)
    X_tr, y_tr, X_te, y_te = X1[tr_m], y1[tr_m], X1[te_m], y1[te_m]

    pred_naive = naive_predict(X_te)
    rf = fit_rf(X_tr, y_tr, seed=CFG["seed"])
    pred_rf = predict_rf(rf, X_te)
    k_val = int(len(X_tr) * 0.85)
    model = CNNLSTM(n_vars=1, window=CFG["window"])
    info = train_torch(model, X_tr[..., None], y_tr, X_tr[k_val:, :, None],
                       y_tr[k_val:], epochs=CFG["epochs"], seed=CFG["seed"])
    pred_cnn = predict_torch(model, X_te[..., None])

    te_dates = dates[tgt_idx[te_m]]
    obs_te = radon_z[tgt_idx[te_m]]
    results = {
        "naive": save_run(out_dir / "naive", "te", obs_te, pred_naive, te_dates),
        "rf": save_run(out_dir / "rf", "te", obs_te, pred_rf, te_dates),
        "cnn_lstm": save_run(out_dir / "cnn_lstm", "te", obs_te, pred_cnn, te_dates,
                             extra={"epochs_run": info["epochs_run"]}),
    }
    print("      指标(RMSE/R2): "
          + ", ".join(f"{k}={v['RMSE']:.3f}/{v['R2']:.3f}" for k, v in results.items()))

    # ---- B: 残差 LOF（主：CNN-LSTM 残差；演示：naive 双变量残差）----
    print("[3/5] 残差 LOF ...")
    res_cnn = np.full(n, np.nan)
    res_cnn[tgt_idx[te_m]] = obs_te - pred_cnn
    d1, sc1, fl1, thr, k = lof_on_residual(res_cnn, dates, tr_i)
    print(f"      k={k}, threshold(98.7%)={thr:.3f}, 异常点={int(fl1.sum())}")

    res_dual = np.full(n, np.nan)
    X2, y2 = make_windows_multi(np.c_[radon_z, level_z], CFG["window"], target_col=0)
    tgt2 = np.arange(CFG["window"], CFG["window"] + len(y2))
    res_dual[tgt2[tr_m]] = y2[tr_m] - naive_predict(X2[tr_m])
    res_dual[tgt2[te_m]] = y2[te_m] - naive_predict(X2[te_m])
    d2, sc2, fl2, thr2, _ = lof_on_residual(res_dual, dates, tr_i)

    print("[4/5] 事件聚合、互证与敏感性 ...")
    events = aggregate_events(d1, sc1, fl1)
    ev_dual = aggregate_events(d2, sc2, fl2)
    if len(events):
        print(events.to_string(index=False))
    sens = quantile_sensitivity(sc1)
    print(sens.to_string(index=False))
    coin = coincidence({"radon_res": events, "dual_res": ev_dual})
    print("      重合统计:\n" + coin.to_string(index=False))
    events.to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")
    ev_dual.to_csv(out_dir / "events_dual.csv", index=False, encoding="utf-8-sig")
    coin.to_csv(out_dir / "coincidence.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out_dir / "sensitivity_quantile.csv", index=False, encoding="utf-8-sig")

    # ---- C: 植入异常检出检验 ----
    print("[5/5] 植入异常检出检验 ...")
    check = []
    for d, lab in planted:
        d = pd.Timestamp(d)
        hit = ((events["start"] - pd.Timedelta(days=7) <= d)
               & (events["end"] + pd.Timedelta(days=7) >= d)) if len(events) else None
        check.append({"planted_date": d.date(), "label": lab,
                      "detected": bool(hit is not None and hit.any())})
    for c in check:
        print(f"      {c['planted_date']} {c['label']}: {'✓ 检出' if c['detected'] else '✗ 漏检'}")

    plot_anomaly_scores(d1, sc1, thr, fl1, out_dir / "anomaly_scores_smoke.png",
                        title="合成数据·CNN-LSTM 残差 LOF 异常得分",
                        extra_events={"planted": pd.DataFrame(
                            [{"start": pd.Timestamp(d)} for d, _ in planted])})
    ok = all(c["detected"] for c in check)
    print(f"\nSMOKE TEST {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
