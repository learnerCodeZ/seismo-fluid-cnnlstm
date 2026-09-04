"""models/lstm_rf.py — LSTM-RF 级联（复现参照文献最优方法）。

文献做法：LSTM 作为特征提取器，其隐藏状态/初步预测输入 RF 做回归校正。
实现：先训练一个小 LSTM 回归器，然后取其**末时刻隐藏状态**作为 RF 的输入特征。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.cnn_lstm import train_torch
from src.models.baselines import fit_rf, predict_rf


class _LSTMExtractor(nn.Module):
    """输出回归值与末时刻隐藏状态。"""

    def __init__(self, n_vars: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_vars, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, return_state: bool = False):
        out, (h, _) = self.lstm(x)
        y = self.head(out[:, -1, :]).squeeze(-1)
        if return_state:
            return y, out[:, -1, :]
        return y


def fit_lstm_rf(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                seed: int = 42, epochs: int = 200, patience: int = 10) -> tuple:
    """训练 LSTM 特征提取器 → 用其末时刻隐藏状态拟合 RF。返回 (lstm, rf)。"""
    n_vars = X.shape[2]
    lstm = _LSTMExtractor(n_vars)
    info = train_torch(lstm, X, y, X_val, y_val, epochs=epochs, patience=patience,
                       seed=seed)
    with torch.no_grad():
        _, H_tr = lstm(torch.tensor(X, dtype=torch.float32), return_state=True)
        _, H_va = lstm(torch.tensor(X_val, dtype=torch.float32), return_state=True)
    # RF 在训练+验证隐藏状态上拟合（隐藏状态已含时序信息，val 不参与梯度）
    H_all = torch.cat([H_tr, H_va]).numpy()
    y_all = np.concatenate([y, y_val])
    rf = fit_rf(H_all, y_all, seed=seed)
    return lstm, rf, info


def predict_lstm_rf(lstm: _LSTMExtractor, rf, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        _, H = lstm(torch.tensor(X, dtype=torch.float32), return_state=True)
    return predict_rf(rf, H.numpy())
