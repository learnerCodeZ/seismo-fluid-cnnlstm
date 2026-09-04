"""viz/figures.py — 统一出图（对齐参照文献图 3–10 版式，300 dpi）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _stamp(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, ls="--", lw=0.5, alpha=0.3)


def plot_prediction(obs, pred, residual, dates, out_path: Path, title: str = "") -> Path:
    """文献图 3–5 版式：上=预测vs观测，下=残差。"""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(dates, obs, lw=0.8, label="观测值")
    a1.plot(dates, pred, lw=0.8, alpha=0.8, label="预测值")
    a1.set_ylabel("value"); a1.legend(loc="upper right"); a1.set_title(title)
    a2.plot(dates, residual, lw=0.7, color="tab:red")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("residual")
    for a in (a1, a2):
        _stamp(a); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    return Path(out_path)


def plot_anomaly_scores(dates, scores, threshold, flags, out_path: Path,
                        title: str = "", extra_events: dict | None = None) -> Path:
    """文献图 6–9 版式：异常得分时序 + 阈值线 + 异常点标注（可叠加多参数垂直线）。"""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, scores, lw=0.7)
    ax.axhline(threshold, color="tab:orange", ls="--", lw=1, label=f"阈值={threshold:.3f}")
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    ax.scatter(d[np.asarray(flags)], np.asarray(scores)[np.asarray(flags)],
               color="tab:red", s=12, zorder=3, label="异常点")
    if extra_events:
        for name, ev in extra_events.items():
            for _, e in ev.iterrows():
                ax.axvline(e["start"], color="gray", ls=":", lw=0.7)
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    _stamp(ax); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    return Path(out_path)
