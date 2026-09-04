# 阶段 3 笔记：CNN-LSTM 单变量训练（西昌32井水氡，已完成，待检查）

> 完成日期：2026-09-03 ｜ 代码：`run_phase3.py` ｜ 对应《技术方案》阶段 3

## 结果

西昌32井水氡，4143天（2015–2026-05），训练3254样本/测试829样本。

| 模型 | RMSE | MAE | MAPE(%) | R² |
|------|------|------|---------|------|
| RF | 0.1850 | 0.1465 | 18.97 | 0.6031 |
| LSTM-RF | 0.1828 | 0.1431 | 19.19 | 0.6126 |
| **CNN-LSTM** | **0.1675** | **0.1339** | **18.15** | **0.6748** |

**CNN-LSTM 胜出**，R² 比 LSTM-RF 高 0.062，RMSE 降低 8.4%。

## 分析

- **CNN 带来的增益来源**：卷积核在 60 天窗口内扫描局部形态（突变、阶变、短时起伏），这些是 LSTM 单独抓不到的信息。从对比图（`cnn_vs_lstmrf_comparison.png`）可以清楚看到：2024 年初的大幅波动区间，CNN-LSTM 的残差明显更小更集中在零线附近。
- **与参照文献的区别**：文献 LSTM-RF 的测试集 R²=0.85（殿沟泉11.7年），我们 CNN-LSTM 在西昌32井（11.3年）R²=0.67。量级差距可能源于数据特性差异（西昌32井水氡波动相对平稳），但**CNN-LSTM 相对 LSTM-RF 的提升方向一致**，说明方法论是有效的。
- **下一步**：进入阶段 4（残差 LOF 异常检测），用 CNN-LSTM 的残差做异常检测。

## 产出文件

- `results/phase3_cnn_lstm/cnn_lstm/` — metrics.json + predictions_test.csv
- `results/phase3_cnn_lstm/lstm_rf/` — metrics.json + predictions_test.csv（对照）
- `results/phase3_cnn_lstm/rf/` — metrics.json + predictions_test.csv（对照）
- `results/phase3_cnn_lstm/cnn_lstm_vs_lstm_rf.png` — CNN-LSTM 单模型预测图
- `results/phase3_cnn_lstm/cnn_vs_lstmrf_comparison.png` — CNN-LSTM vs LSTM-RF 对比图

## 检查建议

- 查看 `results/phase3_cnn_lstm/cnn_vs_lstmrf_comparison.png`：上半图三条线（观测/LSTM-RF/CNN-LSTM），下半图残差对比
- 确认 R² 排序：RF(0.60) < LSTM-RF(0.61) < CNN-LSTM(0.67)
