"""生成一张对比图：观测值 vs CNN-LSTM vs LSTM-RF（干净、可直接展示）。"""
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.common import set_seed, load_paths
set_seed(42)

paths = load_paths()
res_dir = paths["results_dir"]

# 加载 CNN-LSTM 和 LSTM-RF 的预测
cnn = pd.read_csv(res_dir / "window_60" / "cnn_lstm" / "predictions_test.csv", encoding="utf-8-sig")
rf = pd.read_csv(res_dir / "window_60" / "lstm_rf" / "predictions_test.csv", encoding="utf-8-sig")
dates = pd.to_datetime(cnn["date"])
y_true = cnn["y_true"].values
y_cnn = cnn["y_pred"].values
y_rf = rf["y_pred"].values

# 中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [2.5, 1]})

# 上：三条线
a1.plot(dates, y_true, lw=0.8, alpha=0.7, label="观测值", color="#333")
a1.plot(dates, y_rf, lw=1, alpha=0.85, label=f"LSTM-RF (R²=0.6126)", color="#1f77b4", ls="--")
a1.plot(dates, y_cnn, lw=1.2, alpha=0.9, label=f"CNN-LSTM (R²=0.6748)", color="#ff7f0e")
a1.set_ylabel("标准化值", fontsize=12)
a1.legend(fontsize=11, loc="upper right")
a1.set_title("西昌32井水氡浓度·测试段预测对比（窗口=60天）", fontsize=14, fontweight="bold")
a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)
a1.grid(True, ls="--", lw=0.5, alpha=0.3)

# 下：残差对比
a2.plot(dates, y_true - y_rf, lw=0.7, alpha=0.8, label="LSTM-RF 残差", color="#1f77b4")
a2.plot(dates, y_true - y_cnn, lw=0.7, alpha=0.8, label="CNN-LSTM 残差", color="#ff7f0e")
a2.axhline(0, color="k", lw=0.5)
a2.set_ylabel("残差", fontsize=12)
a2.legend(fontsize=9, loc="upper right")
a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
a2.grid(True, ls="--", lw=0.5, alpha=0.3)
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

fig.tight_layout()
out = res_dir / "showcase_cnn_vs_lstmrf.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
