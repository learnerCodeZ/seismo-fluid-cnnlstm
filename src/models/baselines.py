"""models/baselines.py — 三个基线模型（配置对齐参照文献）。

- naive：持久性预测（窗口最后一步 = 预测），作为指标下界；
- RF：100 树、最大深度 100、叶最小 5 样本（文献表3配置）；
- LSTM 单模型：1 层 LSTM(120) + 2 层全连接（文献结构：输入→120→60→30→1）。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def naive_predict(X: np.ndarray) -> np.ndarray:
    """持久性预测：窗口最后一个值即预测。X [N, window] 或 [N, window, n_vars]。"""
    return np.asarray(X)[:, -1, 0] if np.asarray(X).ndim == 3 else np.asarray(X)[:, -1]


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int = 42) -> RandomForestRegressor:
    """文献同款 RF；X 可为 [N, window] 或展平后的 [N, window*n_vars]。"""
    Xf = np.asarray(X).reshape(len(X), -1)
    rf = RandomForestRegressor(n_estimators=100, max_depth=100, min_samples_leaf=5,
                               criterion="squared_error", random_state=seed, n_jobs=-1)
    rf.fit(Xf, np.asarray(y, float))
    return rf


def predict_rf(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return rf.predict(np.asarray(X).reshape(len(X), -1))
